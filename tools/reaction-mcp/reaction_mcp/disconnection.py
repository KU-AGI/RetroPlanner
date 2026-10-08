"""Functional-group + template-compatibility utilities (RDKit-only).

Active helpers backing the ``reaction_*`` tools:
  detect_functional_groups()      -> reaction_detect_functional_groups
  check_template_compatibility()  -> reaction_check_template_compatibility
  check_fg_consistency()          -> internal consistency check

Candidate disconnection sites come from BRICS (propose_disconnections below).
"""
from __future__ import annotations

from rdkit import Chem

# ---------------------------------------------------------------------------
# Functional-group SMARTS catalog
# (name, smarts, fg_type, reactivity_note)
# ---------------------------------------------------------------------------

_FG_CATALOG: list[tuple[str, str, str, str]] = [
    ("primary_alcohol",      "[CH2][OH1]",                         "nucleophilic",         "SN2/oxidation; activate as OMs/OTs"),
    ("secondary_alcohol",    "[CH1X4][OH1]",                       "nucleophilic",         "oxidation to ketone; slower SN2"),
    ("tertiary_alcohol",     "[CX4H0][OH1]",                       "poor_nucleophile",     "E1/SN1; dehydration"),
    ("phenol",               "[OX2H1]c",                           "nucleophilic",         "O-alkylation/acylation; pKa ~10"),
    ("ether",                "[OX2;!$([OH1]);!$(O=*)][#6]",        "inert",                "lone pair donor; acid-labile THP/Bn"),
    ("primary_amine",        "[NX3;H2;!$(Nc=O);!$(NS=O)][#6]",    "nucleophilic",         "alkylation/acylation; pKa ~10"),
    ("secondary_amine",      "[NX3;H1;!$(Nc=O);!$(NS=O)]([#6])[#6]", "nucleophilic",      "acylation/alkylation; weaker base"),
    ("tertiary_amine",       "[NX3;H0;!$(Nc=O)]([#6])([#6])[#6]", "base_catalyst",        "quaternization; catalyst/base"),
    ("aromatic_amine",       "[NX3;H2,H1][cX3]",                  "nucleophilic",         "EAS activator; slower acylation"),
    ("amide",                "[CX3](=O)[NX3]",                     "electrophilic",        "resist nucleophiles; hydrolysis"),
    ("aldehyde",             "[CX3H1](=O)",                        "electrophilic",        "most reactive carbonyl; oxidation"),
    ("ketone",               "[CX3;!H1](=O)([#6])[#6]",           "electrophilic",        "nucleophilic addition; enolization"),
    ("carboxylic_acid",      "[CX3](=O)[OX2H1]",                  "electrophilic",        "esterification/amidation; pKa ~5"),
    ("ester",                "[CX3](=O)[OX2][CX4,c]",             "electrophilic",        "saponification; transesterification"),
    ("lactone",              "[CX3](=O)[OX2][CX4;R]",             "electrophilic",        "ring-open; hydrolysis"),
    ("acid_chloride",        "[CX3](=O)[Cl]",                     "highly_electrophilic", "rapid acylation; moisture-sensitive"),
    ("acid_anhydride",       "[CX3](=O)O[CX3](=O)",               "highly_electrophilic", "acylation; one carboxylate LG"),
    ("alkyl_chloride",       "[CX4][Cl]",                         "electrophilic",        "SN2/E2; moderate LG"),
    ("alkyl_bromide",        "[CX4][Br]",                         "electrophilic",        "SN2/E2; good LG"),
    ("alkyl_iodide",         "[CX4][I]",                          "electrophilic",        "best LG for SN2"),
    ("aryl_halide",          "c[Cl,Br,I]",                        "coupling_partner",     "cross-coupling; SNAr (with EWG)"),
    ("vinyl_halide",         "[CX3]=[CX3][Cl,Br,I]",             "coupling_partner",     "Heck/Suzuki; no SN2"),
    ("terminal_alkene",      "[CH2]=[CH1,CH2]",                   "pi_system",            "addition; metathesis; Wacker"),
    ("internal_alkene",      "[CX3H1]=[CX3H1,CX3H0]",            "pi_system",            "addition; dihydroxylation"),
    ("michael_acceptor",     "[CX3](=O)[CX3]=[CX3]",             "electrophilic",        "conjugate (Michael) addition"),
    ("alkyne",               "[CX2]#[CX2]",                       "pi_system",            "hydrohalogenation; partial reduction"),
    ("diene",                "[CX3]=[CX3]-[CX3]=[CX3]",          "pi_system",            "Diels-Alder 4π component"),
    ("thiol",                "[SX2H1]",                           "soft_nucleophile",     "Michael; disulfide; metal ligation"),
    ("thioether",            "[SX2;!H]([#6])[#6]",               "soft_nucleophile",     "oxidation to sulfoxide/sulfone"),
    ("sulfonyl_ester",       "[SX4](=O)(=O)[OX2][CX4,c]",        "excellent_lg",         "OTs/OMs: best leaving groups"),
    ("boronic_acid",         "[BX3]([OH1])[OH1]",                 "coupling_partner",     "Suzuki coupling; transmetalation"),
    ("boronate_ester",       "[BX3]([OX2])[OX2]",                "coupling_partner",     "Suzuki coupling"),
    ("epoxide",              "[C;R1;r3]1[O;R1;r3][C;R1;r3]1",    "strained_electrophile","ring-opening by Nu"),
    ("nitrile",              "[CX2]#N",                           "electrophilic",        "hydrolysis; reductive amination"),
    ("nitro_group",          "[NX3+](=O)[O-]",                   "ewg",                  "SNAr activator; reduction to amine"),
    ("silyl_ether",          "[SiX4][OX2]",                      "protecting_group",     "TBS/TMS/TIPS; F- deprotection"),
    ("phosphine",            "[PX3]([#6])[#6]",                  "nucleophilic",         "Wittig; Mitsunobu; ligand"),
    # Protecting groups
    ("boc_amine",            "[NX3][CX3](=O)[OX2][CX4]([CH3])([CH3])[CH3]", "protecting_group", "Boc: acid-labile N-PG; remove with TFA"),
    ("cbz_amine",            "[NX3][CX3](=O)[OX2][CH2]c",        "protecting_group",     "Cbz: H2/Pd removal; O-PG alternative"),
    ("bn_ether",             "[OX2][CH2]c",                      "protecting_group",     "Bn ether: O-PG; H2/Pd removal"),
    ("pmb_ether",            "[OX2][CH2]c1ccc(OC)cc1",           "protecting_group",     "PMB ether: DDQ or CAN removal"),
    ("thp_ether",            "[OX2][CH2][CH2][CH2][CH2][O;R]",  "protecting_group",     "THP: acid-labile O-PG"),
    ("acetal",               "[CX4]([OX2][#6])([OX2][#6])[H,#6]", "protecting_group",   "acetal/ketal: aldehyde or ketone protection"),
    # Excellent leaving groups
    ("triflate",             "[OX2][SX4](=O)(=O)C(F)(F)F",      "excellent_lg",         "OTf: best LG; Pd coupling substrate"),
    ("mesylate",             "[OX2][SX4](=O)(=O)[CH3]",         "excellent_lg",         "OMs: SN2/elimination; pair with OTs"),
    # Nitrogen-special
    ("azide",                "[NX2]=[N+]=[N-]",                  "coupling_partner",     "CuAAC click chemistry; reduction to amine"),
    ("isocyanate",           "[NX2]=[CX2]=[OX1]",               "highly_electrophilic", "urea/carbamate; moisture-sensitive"),
    ("isothiocyanate",       "[NX2]=[CX2]=[SX1]",               "highly_electrophilic", "thiourea formation"),
    ("sulfonamide",          "[NX3][SX4](=O)(=O)",               "poor_nucleophile",     "stable N; Mg/MeOH deprotection"),
    ("hydroxylamine",        "[NX3;H1,H2][OX2H1]",              "nucleophilic",         "oxime formation with carbonyl"),
    ("hydrazine",            "[NX3;H1,H2][NX3;H1,H2]",          "nucleophilic",         "hydrazone formation; N-N disconnection"),
    ("n_oxide",              "[N+]([O-])[#6]",                   "ewg",                  "oxidized amine; SNAr activator at o/p"),
    # Acyl-derived
    ("sulfonyl_chloride",    "[SX4](=O)(=O)[Cl]",               "highly_electrophilic", "rapid sulfonamide/ester formation"),
    ("carbamate",            "[NX3][CX3](=O)[OX2][#6]",         "electrophilic",        "urethane; PG for amines; hydrolysis"),
    ("urea",                 "[NX3][CX3](=O)[NX3]",             "electrophilic",        "urea bond; retro = isocyanate + amine"),
    ("imide",                "[NX3]([CX3](=O))[CX3](=O)",       "poor_nucleophile",     "Gabriel synthesis; acidic imide NH"),
    # Michael acceptors
    ("maleimide",            "O=C1CC(=O)[NX3]1",                "highly_electrophilic", "bioconjugation warhead; thiol Michael"),
    ("acrylamide",           "[NX3][CX3](=O)/[CH]=[CH2]",       "electrophilic",        "covalent warhead; radical polymerization"),
    ("vinyl_sulfone",        "[SX4](=O)(=O)/[CH]=[CH2]",        "highly_electrophilic", "irreversible covalent warhead"),
    # Organometallics
    ("grignard",             "[Mg][Cl,Br,I]",                   "nucleophilic",         "strong carbanion equiv; Et2O/THF"),
    ("bpin",                 "[BX3]([OX2]C)([OX2]C)",           "coupling_partner",     "Bpin: Suzuki coupling; stable to air"),
    # Phosphorus
    ("phosphonate_ester",    "[PX4](=O)([OX2][#6])[OX2][#6]",  "ewg",                  "HWE olefination; Michaelis-Arbuzov"),
    ("phosphate_ester",      "[PX4](=O)([OX2])[OX2]",          "excellent_lg",         "good LG in bio; phosphoramidite coupling"),
    # Heteroaromatic N (distinct drug-like reactivity)
    ("pyridine_n",           "n1ccccc1",                         "base_catalyst",        "H-bond acceptor; SNAr at 2/4 position"),
    ("imidazole",            "c1cnc[nH]1",                       "nucleophilic",         "amphoteric; histidine mimic; Cu ligation"),
    # Enol/enamine
    ("enol_ether",           "[CX3]=[CX3][OX2][#6]",            "nucleophilic",         "vinyl O nucleophile; acid-labile"),
    ("enamine",              "[CX3]=[CX3][NX3]",                 "nucleophilic",         "Stork enamine nucleophile (alpha-C equiv)"),
]

_FG_COMPILED: list | None = None

_FG_PRIORITY: dict[str, int] = {
    "highly_electrophilic": 0, "strained_electrophile": 1, "electrophilic": 2,
    "excellent_lg": 3, "coupling_partner": 4, "nucleophilic": 5,
    "soft_nucleophile": 6, "pi_system": 7, "base_catalyst": 8,
    "poor_nucleophile": 9, "ewg": 10, "protecting_group": 11, "inert": 12,
}


def _compile_fg():
    global _FG_COMPILED
    if _FG_COMPILED is None:
        _FG_COMPILED = []
        for name, smarts, fg_type, note in _FG_CATALOG:
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None:
                _FG_COMPILED.append((name, patt, fg_type, note))
    return _FG_COMPILED


def detect_functional_groups(smiles: str) -> list[dict]:
    """Detect named functional groups via SMARTS catalog.

    Returns list of {fg_name, fg_type, atom_indices, primary_atom_idx,
    reactivity_note, heteroatom_adjacency}, sorted by reactivity priority.
    Enables the LLM to reason about disconnection sites from FG context
    rather than raw atom indices.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    results: list[dict] = []
    seen: set[frozenset] = set()

    for name, patt, fg_type, note in _compile_fg():
        for match in mol.GetSubstructMatches(patt):
            key = frozenset(match)
            if key in seen:
                continue
            seen.add(key)
            primary = match[0]
            atom = mol.GetAtomWithIdx(primary)
            hetero_adj = [
                f"{n.GetSymbol()}:{n.GetIdx()}"
                for n in atom.GetNeighbors()
                if n.GetSymbol() not in ("C", "H")
            ]
            results.append({
                "fg_name": name,
                "fg_type": fg_type,
                "atom_indices": list(match),
                "primary_atom_idx": primary,
                "reactivity_note": note,
                "heteroatom_adjacency": hetero_adj,
            })

    results.sort(key=lambda x: (_FG_PRIORITY.get(x["fg_type"], 99), x["primary_atom_idx"]))
    return results


# ---------------------------------------------------------------------------
# Template compatibility check
# ---------------------------------------------------------------------------

def check_template_compatibility(smiles: str, template_smarts: str) -> dict:
    """Check whether a molecule matches a retro SMARTS template.

    template_smarts may be:
      - A plain SMARTS pattern (checked as-is)
      - A retro reaction SMARTS  product>>reactant  (product side is matched)
      - A forward reaction SMARTS  reactant>>product (reactant side is matched)

    Returns {compatible, matched_atoms, match_count, template_class, notes}.
    Useful for template-prior filtering of candidates.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"compatible": False, "error": f"invalid SMILES: {smiles!r}"}

    query_smarts = template_smarts.split(">>")[0].strip() if ">>" in template_smarts else template_smarts
    patt = Chem.MolFromSmarts(query_smarts)
    if patt is None:
        return {"compatible": False, "error": f"invalid SMARTS: {template_smarts!r}"}

    matches = mol.GetSubstructMatches(patt)
    if not matches:
        return {"compatible": False, "matched_atoms": [], "match_count": 0,
                "template_class": "unknown",
                "notes": "molecule does not match SMARTS"}

    # Infer template class from SMARTS content
    ts = query_smarts.lower()
    if any(x in ts for x in ("c[cl", "c[br", "c[i", "[cl,br,i")):
        template_class = "cross_coupling_or_sn2"
    elif "c(=o)" in ts or "c=o" in ts:
        template_class = "acyl_reaction"
    elif "[nh" in ts or "[nx3" in ts:
        template_class = "amine_functionalization"
    elif "c=c" in ts:
        template_class = "alkene_reaction"
    elif "[bx3" in ts or "[b]" in ts:
        template_class = "boronic_acid_coupling"
    elif "[oh" in ts or "[ox2h" in ts:
        template_class = "alcohol_functionalization"
    else:
        template_class = "general"

    return {
        "compatible": True,
        "matched_atoms": [list(m) for m in matches[:5]],
        "match_count": len(matches),
        "template_class": template_class,
        "notes": (f"{len(matches)} match(es); class={template_class}; "
                  f"first match at atoms {list(matches[0])}"),
    }


# ---------------------------------------------------------------------------
# FG consistency check for retro candidates
# ---------------------------------------------------------------------------

def check_fg_consistency(
    bond_changes: list[dict],
    precursor_smiles_list: list[str],
) -> dict:
    """Check whether CGR bond changes are consistent with precursor FGs.

    For each bond formed in the product (i.e. broken retro → formed forward),
    verifies the precursors contain the expected nucleophile + electrophile pair.
    E.g.: C-N bond formed → precursors should have N nucleophile + electrophilic C.

    bond_changes: from cgr.analyze() output (bonds_formed list)
    precursor_smiles_list: list of precursor SMILES strings

    Returns {consistent, score (0-1), issues, matched_bonds}.
    """
    if not bond_changes or not precursor_smiles_list:
        return {"consistent": None, "score": None, "issues": ["no data to check"]}

    all_fgs: list[dict] = []
    for smi in precursor_smiles_list:
        all_fgs.extend(detect_functional_groups(smi))

    nucleophilic_types = {"nucleophilic", "soft_nucleophile"}
    electrophilic_types = {"electrophilic", "highly_electrophilic", "strained_electrophile", "excellent_lg"}

    has_nu = any(fg["fg_type"] in nucleophilic_types for fg in all_fgs)
    has_el = any(fg["fg_type"] in electrophilic_types for fg in all_fgs)

    issues: list[str] = []
    matched: list[str] = []

    for bc in bond_changes[:3]:  # check top 3 bond changes
        atoms_str = bc.get("atoms", "?")
        a1_fg = bc.get("atom1_fg", "")
        a2_fg = bc.get("atom2_fg", "")

        # C-N bond formed: expect N nucleophile + electrophilic C
        if "C-N" in atoms_str or "N-C" in atoms_str:
            if has_nu and has_el:
                matched.append(f"C-N bond: Nu+El present ✓")
            elif not has_nu:
                issues.append(f"C-N bond formed but no nucleophilic N in precursors")
            else:
                issues.append(f"C-N bond formed but no electrophilic C in precursors")

        # C-O bond formed
        elif "C-O" in atoms_str or "O-C" in atoms_str:
            if has_nu and has_el:
                matched.append(f"C-O bond: Nu+El present ✓")
            else:
                issues.append(f"C-O bond formed but missing Nu or El in precursors")

        # C-C bond formed: expect enolate or organometallic + electrophilic C
        elif atoms_str == "C-C":
            alpha_co_present = any(
                "alpha" in fg.get("fg_name", "") or "enol" in fg.get("reactivity_note", "")
                for fg in all_fgs
            )
            if alpha_co_present or has_el:
                matched.append(f"C-C bond: reactive C present ✓")
            else:
                issues.append("C-C bond formed but no alpha-carbonyl or electrophilic C in precursors")

        else:
            matched.append(f"{atoms_str} bond: assumed plausible")

    total = len(bond_changes[:3])
    n_matched = len(matched)
    consistency_score = n_matched / total if total > 0 else 0.5

    return {
        "consistent": len(issues) == 0,
        "score": round(consistency_score, 2),
        "issues": issues,
        "matched_bonds": matched,
        "precursor_fg_types": list({fg["fg_type"] for fg in all_fgs}),
    }


# ---------------------------------------------------------------------------
# BRICS disconnection proposal
# ---------------------------------------------------------------------------
# BRICS ("Breaking of Retrosynthetically Interesting Chemical Substructures",
# Degen et al. ChemMedChem 2008) cleaves bonds at 16 reaction-derived atom
# environments (L1..L16). We expose each cleavable bond as a *candidate*
# disconnection — the retrosynthetically-motivated bond + the two synthons it
# yields + a rule-based bond type — and leave feasibility to the learned retro
# model. BRICS is context-blind (it cannot judge whether the specific substrate
# actually reacts, and it never cuts fused rings), so this PROPOSES leads; it
# does not decide. Confirm every candidate with reaction_predict_singlestep_retro.

# Semantic label per BRICS L-type atom environment (from BRICS.environs SMARTS).
_BRICS_L_LABEL = {
    "1": "carbonyl-C", "3": "O", "4": "alkyl-C", "5": "amine-N",
    "6": "acyl-C", "7": "alkene-C", "8": "alkyl-C", "9": "aromatic-N",
    "10": "ring-amide-N", "11": "thioether-S", "12": "sulfonyl-S",
    "13": "ring-C", "14": "heteroaromatic-C", "15": "ring-C", "16": "aryl-C",
}
# Recognizable named bonds for common L-pairs (order-insensitive).
_BRICS_PAIR_NAME = {
    frozenset(("1", "3")): "ester",
    frozenset(("1", "5")): "amide",
    frozenset(("1", "10")): "amide (acyl–ring N)",
    frozenset(("3", "4")): "ether (C–O)",
    frozenset(("3", "13")): "aryl/ring ether",
    frozenset(("3", "14")): "aryl ether",
    frozenset(("3", "16")): "aryl ether",
    frozenset(("4", "5")): "amine (C–N)",
    frozenset(("4", "11")): "thioether (C–S)",
    frozenset(("5", "12")): "sulfonamide",
    frozenset(("5", "13")): "N-aryl amine",
    frozenset(("5", "14")): "N-aryl amine",
    frozenset(("5", "16")): "N-aryl amine (Buchwald/Ullmann)",
    frozenset(("6", "16")): "aryl ketone / acyl–aryl",
    frozenset(("7", "7")): "olefin (C=C)",
    frozenset(("16", "16")): "biaryl (Ar–Ar, Suzuki-type)",
    frozenset(("13", "16")): "biaryl / ring–aryl",
    frozenset(("14", "16")): "biaryl",
}


def _brics_bond_type(l1: str, l2: str) -> str:
    """Human label for a BRICS cleavage from its L-type pair (rule-based lookup)."""
    n1 = "".join(c for c in l1 if c.isdigit())
    n2 = "".join(c for c in l2 if c.isdigit())
    named = _BRICS_PAIR_NAME.get(frozenset((n1, n2)))
    if named:
        return named
    a = _BRICS_L_LABEL.get(n1, f"L{n1}")
    b = _BRICS_L_LABEL.get(n2, f"L{n2}")
    return f"{a}–{b}"


def _heavy_atom_count(smiles: str) -> int:
    m = Chem.MolFromSmiles(smiles)
    return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 0) if m else 0


def propose_disconnections(smiles: str) -> dict:
    """Propose candidate retro disconnections for a molecule via BRICS bonds.

    For each bond BRICS would cleave, returns the bond, a rule-based ``bond_type``
    (from the BRICS L-type pair), the two ``synthons`` produced by cutting *only*
    that bond (dummy ``[*]`` marks the attachment point), and their factual
    ``synthon_sizes`` (heavy-atom counts). These are candidates, NOT verified
    disconnections — BRICS is context-blind and never cuts fused rings, so
    confirm each with ``reaction_predict_singlestep_retro``.

    Returns ``{smiles, n, disconnections: [...]}`` sorted so balanced (more
    convergent-looking) cuts come first — but no convergence judgment is imposed;
    the raw sizes are provided for the caller to reason over.
    """
    from rdkit.Chem import BRICS

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"Invalid SMILES: {smiles!r}", "valid": False}

    out: list[dict] = []
    for (a1, a2), (l1, l2) in BRICS.FindBRICSBonds(mol):
        bond = mol.GetBondBetweenAtoms(a1, a2)
        frag = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True)
        synthons = Chem.MolToSmiles(frag).split(".")
        sizes = sorted(_heavy_atom_count(s) for s in synthons)
        out.append({
            "bond": f"{mol.GetAtomWithIdx(a1).GetSymbol()}{a1}-"
                    f"{mol.GetAtomWithIdx(a2).GetSymbol()}{a2}",
            "bond_type": _brics_bond_type(l1, l2),
            "synthons": synthons,
            "synthon_sizes": sizes,
        })

    # balanced cuts first (largest min-fragment), purely as ordering — not a flag
    out.sort(key=lambda d: min(d["synthon_sizes"]) if d["synthon_sizes"] else 0,
             reverse=True)
    return {"smiles": smiles, "n": len(out), "disconnections": out,
            "valid": True, "error": None}
