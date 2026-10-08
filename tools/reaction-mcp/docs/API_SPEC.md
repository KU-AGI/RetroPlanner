# reaction-mcp API specification

`reaction_mcp/server.py` exposes **11** functions via `@mcp.tool`. The MCP is scoped to single-step retrosynthesis and reaction analysis: there is no multi-step, forward, condition or mechanism tool. The single-step models are R-SMILES and LocalRetro.

Every tool takes a common `return_text: bool = False` argument. `False` returns a structured `dict`; `True` returns a human-readable one-line summary `str`. The examples below show the `dict` form.

## At a glance

| # | Tool | Role (one line) | Core package / backend | Input → output |
|---|------|-------------|---------------------|-------------|
| 1 | `reaction_atom_map_reaction` | Assign atom mapping to a reaction SMILES | `rxnmapper` (IBM transformer) | `R>>P` → `{mapped_rxn, confidence}` |
| 2 | `reaction_predict_singlestep_retro` | Single-step retrosynthesis (two-model pool + cross-model voting) | `pool/`: R-SMILES (`rsmiles`) · LocalRetro (`localretro`) | `product_smiles` → ranked list of precursor candidates |
| 3 | `reaction_get_bond_changes` | Extract net bond changes of a reaction (single/batch) | `CGRtools` + `rxnmapper` | `R>>P` or `product+candidates` → formed/broken bonds + reaction type |
| 4 | `reaction_detect_functional_groups` | Detect functional groups in a molecule + reactivity labels | `RDKit` (SMARTS catalog) | `smiles` → functional group list (type/atom indices/reactivity note) |
| 5 | `reaction_match_templates` | Find named reactions that can make a molecule (no hand-written SMARTS needed) | `RDKit` + Rxn-INSIGHT named SMIRKS corpus (~470) | `smiles` → applicable named reactions + SMIRKS |
| 6 | `reaction_check_template_compatibility` | Verify compatibility of a molecule with a given SMARTS template | `RDKit` | `smiles` + `template_smarts` → `{compatible, matched_atoms, …}` |
| 7 | `reaction_extract_reaction_center` | Reaction-center atoms/bonds + atom-level changes + graded local environment (radius 1..N) | `CGRtools` + `rxnmapper` | `R>>P` (+`max_radius`) → center_atoms/bonds + radius-0 template + local_environment |
| 8 | `reaction_extract_reaction_template` | Extract a generalized retro template (SMARTS) from a reaction | `rdchiral` + `rxnmapper` | `R>>P` → `{retro_template(product>>reactants), necessary_reagent, intra_only, dimer_only}` |
| 9 | `reaction_propose_disconnections` | Propose BRICS disconnection candidate bonds (hint-only; retro does the verification) | `RDKit` (BRICS) | `product_smiles` → `[{bond, bond_type, synthons, synthon_sizes}]` |
| 10 | `reaction_route_target` | Decide single-step vs multi-step at a node (does one step reach stock?) | `pool/` + stock + complexity prior | `smiles` → `{decision, basis, chosen_disconnection, …}` |
| 11 | `reaction_in_stock` | Purchasable-building-block lookup | configured stock (`REACTION_STOCK`) | `smiles` (str or list) → in-stock flags |

---

## Detailed specification

### 1. `reaction_atom_map_reaction`
- **Role**: Map reactant–product atom correspondence of a reaction SMILES. (Tools 3 and 7 map automatically internally, so call this directly only when you need the mapping itself.)
- **Package**: `rxnmapper`
- **Input**
  | Argument | Type | Default | Description |
  |---|---|---|---|
  | `reaction_smiles` | `str` | (required) | `reactants>>product` |
  | `return_text` | `bool` | `False` | Whether to return a text summary |
- **Output example**
  ```jsonc
  // in:  "CC(=O)O.CCO>>CC(=O)OCC"
  {"mapped_rxn": "[CH3:1][C:2](=[O:3])[OH:4].[CH3:5][CH2:6][OH:7]>>[CH3:1][C:2](=[O:3])[O:7][CH2:6][CH3:5]",
   "confidence": 0.98}
  ```

### 2. `reaction_predict_singlestep_retro`
- **Role**: Single-step retrosynthesis. Fans out to the live retro backends → drops precursor sets that fail the mass guard → aggregates cross-model votes under the selection backbone. `candidates` is the primary output for downstream analysis.
- **Package / backend**: `pool/` — R-SMILES (`rsmiles`, HTTP :8100) and LocalRetro (`localretro`, HTTP :8097). No forward backend is configured, so round-trip verification only marks candidates `round_trip.matched = null`.
- **Input**
  | Argument | Type | Default | Description |
  |---|---|---|---|
  | `product_smiles` | `str` | (required) | Target product |
  | `top_k` | `int` | `10` | Number of candidates to return |
  | `models` | `list[str]\|None` | `None` | Use only specific backends (`rsmiles`, `localretro`) |
  | `round_trip` | `bool` | `False` | Forward round-trip verification (a no-op without a forward backend) |
  - Server-side env overrides: `REACTION_RETRO_SELECTION` (consensus/single/plurality/rrf/union), `REACTION_RETRO_SINGLE_MODEL` (default `rsmiles`), `REACTION_RETRO_VERIFY` (off/soft/hard), `REACTION_RETRO_GATE_TAU`, `REACTION_RETRO_DEPTH`, `REACTION_RETRO_TOPK`.
- **Output example**
  ```jsonc
  // in:  product_smiles="CC(=O)OCC"
  {"task": "retro", "product": "CC(=O)OCC",
   "selection": "consensus", "verify": "off",
   "agreement": 2, "mass_rejected": 0, "gate_tau": 0, "abstained": false,
   "candidates": [
     {"molecules": ["CC(=O)O", "CCO"], "n_backends": 2, "backends": ["localretro", "rsmiles"],
      "consensus_score": 3.0, "best_rank": 0},
     {"molecules": ["CC(=O)Cl", "CCO"], "n_backends": 1, "backends": ["rsmiles"],
      "consensus_score": 1.5, "best_rank": 1}
   ],
   "top1": {"molecules": ["CC(=O)O", "CCO"], "...": "..."},
   "backends_ok": ["localretro", "rsmiles"], "backends_failed": []}
  ```

### 3. `reaction_get_bond_changes`
- **Role**: *Extract* the net bond changes of a reaction via CGR (no feasibility judgment). Single mode (`reaction_smiles`) or batch retro mode (`product_smiles`+`candidates`). Mapping is automatic.
- **Package**: `CGRtools` + `rxnmapper`
- **Input**
  | Argument | Type | Default | Description |
  |---|---|---|---|
  | `reaction_smiles` | `str\|None` | `None` | Single mode: `R>>P` |
  | `product_smiles` | `str\|None` | `None` | Batch mode: target product |
  | `candidates` | `list\|None` | `None` | Batch mode: precursor sets (`"A.B"` / `["A","B"]` / `{"molecules":[...]}`) |
  - Specify exactly one of single or batch.
- **Output example (single)**
  ```jsonc
  // in:  reaction_smiles="CC(=O)O.CCO>>CC(=O)OCC"
  {"valid": true, "reaction_type": "esterification",
   "bonds_formed": [{"atoms": ["C","O"], "bond_order": 1, "atom1_fg": "carbonyl", "atom2_fg": "hydroxyl"}],
   "bonds_broken": [{"atoms": ["O","H"], "bond_order": 1}],
   "bonds_order_changed": [],
   "n_bonds_formed": 1, "n_bonds_broken": 1,
   "charge_changes": [], "cgr_summary": "...", "mapping_confidence": 0.97}
  ```
- **Output example (batch)**: `{"product": ..., "results": [{candidate_idx, candidate_smiles, valid, reaction_type, ...}], "n": N, "n_valid": M}`

### 4. `reaction_detect_functional_groups`
- **Role**: Detect named functional groups in a molecule with a SMARTS catalog and assign reactivity-priority labels. Used to reason about which bonds are disconnection candidates and which named reactions are possible.
- **Package**: `RDKit`
- **Input**
  | Argument | Type | Default | Description |
  |---|---|---|---|
  | `smiles` | `str` | (required) | Molecule SMILES |
- **Output example**
  ```jsonc
  // in:  smiles="CC(=O)Cl"
  {"smiles": "CC(=O)Cl", "n": 1,
   "functional_groups": [
     {"fg_name": "acyl_chloride", "fg_type": "highly_electrophilic",
      "atom_indices": [1,2,3], "primary_atom_idx": 1,
      "reactivity_note": "strong acylating agent", "heteroatom_adjacency": true}]}
  ```
  - `fg_type` priority: highly_electrophilic, strained_electrophile, electrophilic, excellent_lg, coupling_partner, nucleophilic, soft_nucleophile, pi_system, base_catalyst, poor_nucleophile, ewg, protecting_group, inert.

### 5. `reaction_match_templates`
- **Role**: Match *named reactions* from the corpus that can produce the given molecule (no hand-written SMARTS needed). Ranked by match specificity. Each result's `template` SMIRKS can be passed as-is to tool 6.
- **Package**: `RDKit` + Rxn-INSIGHT named SMIRKS corpus (`data/named_reaction_smirks.json`, ~470 entries). The corpus can be swapped via `REACTION_TEMPLATE_LIBRARY`.
- **Input**
  | Argument | Type | Default | Description |
  |---|---|---|---|
  | `smiles` | `str` | (required) | Target molecule |
  | `top_k` | `int` | `10` | Number to return |
  | `name_filter` | `str\|None` | `None` | Name substring filter (case-insensitive) |
- **Output example**
  ```jsonc
  // in:  smiles="c1ccc(-c2ccccc2)cc1"
  {"valid": true, "n_applicable": 3,
   "names": ["Suzuki coupling with boronic acids", "Negishi coupling", "Stille reaction"],
   "templates": [
     {"name": "Suzuki coupling with boronic acids",
      "template": "[c:1]-[c:2]>>[c:1]-[Br].[c:2]-B(O)O",
      "matched_atoms": [...], "specificity": 0.88}]}
  ```

### 6. `reaction_check_template_compatibility`
- **Role**: Verify in detail whether a molecule matches a given retro/forward SMARTS template. Used to verify a `template` obtained from tool 5.
- **Package**: `RDKit`
- **Input**
  | Argument | Type | Default | Description |
  |---|---|---|---|
  | `smiles` | `str` | (required) | Molecule |
  | `template_smarts` | `str` | (required) | plain SMARTS / `product>>reactant` (retro) / `reactant>>product` (forward) |
- **Output example**
  ```jsonc
  // in:  smiles="CCBr", template_smarts="[CX4][Br]"
  {"compatible": true, "matched_atoms": [[1,2]], "match_count": 1,
   "template_class": "alkyl_halide", "notes": "..."}
  ```

### 7. `reaction_extract_reaction_center`
- **Role**: Extract the reaction center, and also report the *atom-level* changes of every changing atom (rehybridization/neighbor count/charge), the radius-0 template, and the **radius 1..`max_radius` graded local environment** (reaction center + bond shells). One level more detailed than tool 3. Radius-0 alone misses the steric/electronic environment that governs template applicability, so the shells are provided alongside.
- **Package**: `CGRtools` + `rxnmapper` (automatic mapping)
- **Input**
  | Argument | Type | Default | Description |
  |---|---|---|---|
  | `reaction_smiles` | `str` | (required) | `R>>P` |
  | `max_radius` | `int` | `2` | Local environment expansion radius (capped at 3; 0 means radius-0 only) |
- **Output example**
  ```jsonc
  // in:  reaction_smiles="CC(=O)O.CCO>>CC(=O)OCC", max_radius=2
  {"valid": true, "reaction_type": "esterification",
   "center_atoms": [
     {"element": "C", "fg": "carbonyl", "hybridization": "sp2",
      "hybridization_change": null, "neighbor_change": 0, "charge_change": 0, "in_ring": false}],
   "center_bonds": [{"atoms": ["C","O"], "change": "formed", "order": 1}],
   "center_template": "O[.>-]C[->.]O",
   "local_environment": [
     {"radius": 1, "n_atoms": 6, "template": "O=C([.>-]OC)(C)[->.]O"},
     {"radius": 2, "n_atoms": 7, "template": "C(=O)([.>-]OCC)(C)[->.]O"}],
   "n_components": 1, "mapping_confidence": 0.97}
  ```

### 8. `reaction_extract_reaction_template`
- **Role**: **Extract a generalized retro template (SMARTS) on the fly** from the given reaction. Unlike fixed named-corpus matching (#5), it builds a reaction-specific template for an arbitrary reaction, and the result can be fed as-is to #6 for verification. rdchiral abstracts the changing atoms (leaving/attacking group, H count, degree, charge) into a reusable `product>>reactants` SMARTS.
- **Package**: `rdchiral` + `rxnmapper` (automatic mapping)
- **Input**
  | Argument | Type | Default | Description |
  |---|---|---|---|
  | `reaction_smiles` | `str` | (required) | `reactants>>product` |
- **Output example**
  ```jsonc
  // in:  reaction_smiles="CC(=O)O.CCO>>CC(=O)OCC"
  {"valid": true,
   "retro_template": "[C:5]-[O;H0;D2;+0:6]-[C;H0;D3;+0:2](-[C;D1;H3:1])=[O;D1;H0:3]>>[C;D1;H3:1]-[C;H0;D3;+0:2](=[O;D1;H0:3])-[OH;D1;+0:4].[C:5]-[OH;D1;+0:6]",
   "necessary_reagent": null, "intra_only": false, "dimer_only": false,
   "mapping_confidence": 0.97}
  ```

### 9. `reaction_propose_disconnections`
- **Role**: **Propose disconnection candidate bonds** of the target with BRICS (16 reaction-derived rules). Each candidate has a rule-based `bond_type` (L-pair lookup) + the `synthons` from cutting only that bond + `synthon_sizes` (heavy-atom counts). **A hint generator, not a decider** — BRICS is context-blind and cannot cut fused rings, so each candidate needs a feasibility check with `reaction_predict_singlestep_retro`/`reaction_match_templates`. No disconnection candidates does not mean unsynthesizable (ring-forming disconnections are not visible). Building-block/stock matching is intentionally excluded.
- **Package**: `RDKit` (BRICS)
- **Input**
  | Argument | Type | Default | Description |
  |---|---|---|---|
  | `product_smiles` | `str` | (required) | Target molecule |
- **Output example**
  ```jsonc
  // in:  product_smiles="CCN(CC)CCNC(=O)c1ccc(N)cc1"
  {"smiles": "CCN(CC)CCNC(=O)c1ccc(N)cc1", "n": 6,
   "disconnections": [
     {"bond": "C8-N7", "bond_type": "amide",
      "synthons": ["[*]C(=O)c1ccc(N)cc1", "[*]NCCN(CC)CC"], "synthon_sizes": [8, 9]},
     {"bond": "C6-N7", "bond_type": "amine (C–N)",
      "synthons": ["[*]NC(=O)c1ccc(N)cc1", "[*]CCN(CC)CC"], "synthon_sizes": [7, 10]}
     // ... sorted so balanced (convergent) disconnections come first (no judgment flag)
   ]}
  ```

---

## Note: model pool (`reaction_mcp/pool/`)

`reaction_predict_singlestep_retro` and `reaction_route_target` are served by a two-model pool: R-SMILES and LocalRetro. Each model is wrapped as a `Predictor` that advertises the `Task` it serves; `ModelPool` fans requests out to all live backends, and the `consensus` layer aggregates candidates into one ranked list by cross-model voting. Each backend runs as a warm HTTP microservice in its own conda env and is discovered via env vars. (See `README.md` for the backend table and setup.)
