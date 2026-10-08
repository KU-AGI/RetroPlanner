"""reaction-mcp: reaction prediction and analysis tools.

Active tools (11):
  Prediction pool  : atom_map_reaction, predict_singlestep_retro (R-SMILES +
                     LocalRetro consensus), route_target (single-step vs
                     multi-step advisor)
  Bond changes     : get_bond_changes
  Disconnection    : propose_disconnections (BRICS candidate sites; hint-only)
  FG + templates   : detect_functional_groups, match_templates,
                     check_template_compatibility, extract_reaction_template
  CGR topology     : extract_reaction_center (radius-0 mechanism + local environment)
  Stock            : in_stock

The single-step retro models are the two warm servers configured in
``pool/config.py`` (rsmiles, localretro). There is no forward, multi-step or
condition tool; multi-step route search runs outside the MCP server
(``scripts/traj_route_search.py`` over the SSR fleets).
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from fastmcp import FastMCP
from rdkit import Chem

_TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

from common import (  # noqa: E402
    ToolErrorType,
    error_envelope,
    format_output,
    register_template,
)

from .cgr import (  # noqa: E402
    analyze as _cgr_analyze,
    analyze_candidates as _cgr_analyze_candidates,
    extract_reaction_center as _cgr_extract_center,
)
from .pool import Task, aggregate, get_pool, round_trip_filter  # noqa: E402
from .disconnection import (  # noqa: E402
    check_template_compatibility as _check_template_compat,
    detect_functional_groups as _detect_fg,
    propose_disconnections as _propose_disconnections,
)
from .templates import (  # noqa: E402
    match_templates as _match_templates,
    extract_reaction_template as _extract_reaction_template,
)
from .stock import check_one as _stock_check_one, stock_status as _stock_status  # noqa: E402

mcp = FastMCP("reaction-mcp")

_RXN_MAPPER = None
_RXN_MAPPER_LOCK = threading.Lock()


def _get_rxnmapper():
    global _RXN_MAPPER
    if _RXN_MAPPER is None:
        with _RXN_MAPPER_LOCK:
            if _RXN_MAPPER is None:
                from rxnmapper import RXNMapper
                _RXN_MAPPER = RXNMapper()
    return _RXN_MAPPER


def _ensure_mapped(reaction_smiles: str) -> tuple[str | None, float | None, str | None]:
    """Atom-map a reaction SMILES via RXNMapper if not already mapped.

    Returns ``(mapped_smiles, confidence, error)``. If the input already carries
    atom maps (contains ``:``) it is returned unchanged with no confidence.
    """
    if ":" in reaction_smiles:
        return reaction_smiles, None, None
    try:
        res = _get_rxnmapper().get_attention_guided_atom_maps([reaction_smiles])
    except Exception as e:
        return None, None, f"RXNMapper failed: {e}. Pass an atom-mapped SMILES to bypass."
    if not res:
        return None, None, "RXNMapper returned no mapping"
    return res[0].get("mapped_rxn", reaction_smiles), float(res[0].get("confidence", 0.0)), None


def _parse_rxn(reaction_smiles: str) -> tuple[list, list] | None:
    if ">>" not in reaction_smiles:
        return None
    left, right = reaction_smiles.split(">>", 1)
    if ">" in left:
        left = left.split(">")[0]
    reactants = [Chem.MolFromSmiles(s) for s in left.split(".") if s]
    products = [Chem.MolFromSmiles(s) for s in right.split(".") if s]
    if any(m is None for m in reactants + products):
        return None
    return reactants, products


def _compact(candidates: list[dict], n: int = 5) -> list[str]:
    out = []
    for c in candidates[:n]:
        mols = "+".join(c.get("molecules", []))
        rt = c.get("round_trip", {})
        rt_tag = " rt=yes" if rt.get("matched") else " rt=no" if rt else ""
        out.append(
            f"{mols} (votes={c.get('votes')} backends={','.join(c.get('backends', []))}{rt_tag})"
        )
    return out


# ===========================================================================
# Prediction pool tools
# ===========================================================================

_TPL_ATOM_MAP = register_template(
    "reaction_atom_map_reaction",
    "Atom-mapped the reaction (mapper confidence {confidence}). Mapped SMILES: {mapped_rxn}",
)


@mcp.tool
def reaction_atom_map_reaction(
    reaction_smiles: str, return_text: bool = False
) -> dict | str:
    """Atom-map a reaction SMILES via RXNMapper (IBM transformer).

    Mapping is applied automatically inside reaction_get_bond_changes /
    reaction_extract_reaction_center; call this directly only when you want the
    mapped SMILES itself. Assigns correspondence between reactant and product
    atoms, enabling bond-change extraction.
    """
    parsed = _parse_rxn(reaction_smiles)
    if parsed is None:
        return format_output(
            error_envelope(f"Invalid reaction SMILES: {reaction_smiles!r}", ToolErrorType.PARSE_ERROR),
            _TPL_ATOM_MAP, return_text, tool_name="reaction_atom_map_reaction",
        )
    try:
        mapper = _get_rxnmapper()
        results = mapper.get_attention_guided_atom_maps([reaction_smiles])
    except Exception as e:
        msg = str(e)
        error_type = (
            ToolErrorType.UNSUPPORTED if "CUDA error" in msg or "no kernel image" in msg
            else ToolErrorType.INTERNAL_ERROR
        )
        return format_output(
            error_envelope(msg, error_type),
            _TPL_ATOM_MAP, return_text, tool_name="reaction_atom_map_reaction",
        )
    if not results:
        return format_output(
            error_envelope("rxnmapper returned no mapping", ToolErrorType.UPSTREAM_ERROR),
            _TPL_ATOM_MAP, return_text, tool_name="reaction_atom_map_reaction",
        )
    res = results[0]
    return format_output(
        {"mapped_rxn": res.get("mapped_rxn"), "confidence": float(res.get("confidence", 0.0))},
        _TPL_ATOM_MAP, return_text, tool_name="reaction_atom_map_reaction",
    )


_TPL_POOL_RETRO = register_template(
    "reaction_predict_singlestep_retro",
    "Retrosynthesis (live backends: {backends_ok}). Best precursors: {top}. Ranked precursor sets: {candidates}",
)


def _retro_selection_config() -> dict:
    """Retro candidate-selection mode, read from env at call time (deploy-friendly).

    Set these BEFORE launching the reaction server (``python -m reaction_mcp``)
    to fix how every retro call
    combines the per-model candidate lists. See pool/consensus.py.

        REACTION_RETRO_SELECTION  consensus | single | plurality | rrf | union
                                  (default: consensus = approval vote + MRR)
        REACTION_RETRO_SINGLE_MODEL  for selection=single, which model is the
                                  single (default: rsmiles). FAIL-CLOSED: if that
                                  model did not respond, the call ABSTAINS rather
                                  than silently using a weaker live model. Set to
                                  "auto" for legacy best-available behaviour.
        REACTION_RETRO_VERIFY     off | soft | hard   forward round-trip use
                                  (default: unset -> follows the round_trip arg;
                                  a no-op while no forward backend is configured)
        REACTION_RETRO_GATE_TAU   int; >0 abstains when fewer than TAU models
                                  agree on their top-1 (default: 0 = off)
        REACTION_RETRO_DEPTH      int; per-model candidate depth fed into the
                                  pool (default: unset -> = final top_k). Widens
                                  the union / fusion pool per model.
        REACTION_RETRO_TOPK       int; final number of candidates returned,
                                  SEPARATE from per-model depth. Overrides the
                                  tool's top_k arg when set (server-fixed).
    """
    def _int_env(name):
        v = os.getenv(name)
        if not v:
            return None
        try:
            return int(v)
        except ValueError:
            return None

    backbone = (os.getenv("REACTION_RETRO_SELECTION") or "consensus").strip().lower()
    verify_env = os.getenv("REACTION_RETRO_VERIFY")
    verify = verify_env.strip().lower() if verify_env else None
    # For backbone=single, pin WHICH model is the single. Default rsmiles (the
    # stronger of the two on USPTO-50K top-1) so `single` is fail-closed: if
    # rsmiles did not respond the call ABSTAINS instead of silently using
    # localretro. Set REACTION_RETRO_SINGLE_MODEL=auto for best-available.
    single_model = (os.getenv("REACTION_RETRO_SINGLE_MODEL") or "rsmiles").strip().lower()
    return {
        "backbone": backbone,
        "verify": verify,
        "gate_tau": _int_env("REACTION_RETRO_GATE_TAU") or 0,
        "depth": _int_env("REACTION_RETRO_DEPTH"),
        "topk": _int_env("REACTION_RETRO_TOPK"),
        "single_model": single_model,
    }


@mcp.tool
def reaction_predict_singlestep_retro(
    product_smiles: str,
    top_k: int = 10,
    models: list[str] | None = None,
    round_trip: bool = False,
    return_text: bool = False,
) -> dict | str:
    """Single-step retrosynthesis across the model pool (R-SMILES + LocalRetro).

    Fans out to every live single-step retro backend and aggregates by
    cross-model votes under the configured selection backbone
    (``REACTION_RETRO_SELECTION``; ``union`` returns the pooled pass@k upper
    bound). To decide whether one step is enough, consult ``reaction_route_target``.

    Round-trip verification is OFF by default (``round_trip=False``). It re-ranks
    candidates by forward-pool consistency, but no forward backend is configured
    in this pool, so turning it on (``round_trip=True`` or
    ``REACTION_RETRO_VERIFY=soft|hard``; the env wins over the arg) only marks
    each candidate ``round_trip.matched = None`` and leaves the order unchanged.

    The ``candidates`` list is the primary output for downstream analysis —
    pass it (with ``product_smiles``) to ``reaction_get_bond_changes`` to extract
    per-candidate bond-change info.
    """
    if not product_smiles:
        return format_output(
            error_envelope("product_smiles is empty", ToolErrorType.INVALID_INPUT),
            _TPL_POOL_RETRO, return_text, tool_name="reaction_predict_singlestep_retro",
        )
    pool = get_pool()
    sel = _retro_selection_config()
    # depth = per-model fan-out (how many each backend returns into the pool);
    # final_k = how many consensus candidates are returned. Both default to the
    # tool's top_k arg unless fixed server-side via REACTION_RETRO_DEPTH/_TOPK.
    depth = sel["depth"] or top_k
    final_k = sel["topk"] or top_k
    req = {"product": product_smiles, "target": product_smiles, "top_k": depth}
    results = pool.run(Task.RETRO, req, models=models)
    if not results:
        return format_output(
            error_envelope("no live retro backend", ToolErrorType.UNSUPPORTED),
            _TPL_POOL_RETRO, return_text, tool_name="reaction_predict_singlestep_retro",
        )
    agg = aggregate(results, top_k=final_k,
                    backbone=sel["backbone"], gate_tau=sel["gate_tau"],
                    single_model=sel["single_model"],
                    product=product_smiles)   # enables the mass guard
    cands = agg["candidates"]
    # verify (forward round-trip): env REACTION_RETRO_VERIFY wins; else the
    # round_trip arg maps to soft (on) / off.
    verify = sel["verify"] if sel["verify"] is not None else ("soft" if round_trip else "off")
    if verify != "off" and cands:
        try:
            cands = round_trip_filter(pool, cands, product_smiles, mode=verify)
        except Exception:
            pass
    return format_output(
        {
            "task": Task.RETRO.value,
            "product": product_smiles,
            "selection": agg["selection"],
            "single_required": agg["single_required"],
            "single_effective": agg["single_effective"],
            "verify": verify,
            "agreement": agg["agreement"],
            "mass_rejected": agg.get("mass_rejected", 0),
            "gate_tau": agg["gate_tau"],
            "abstained": agg["abstained"],
            "candidates": cands,
            "top1": None if agg["abstained"] else (cands[0] if cands else None),
            "backends_ok": agg["backends_ok"],
            "backends_failed": agg["backends_failed"],
        },
        _TPL_POOL_RETRO, return_text, tool_name="reaction_predict_singlestep_retro",
        text_payload={"backends_ok": agg["backends_ok"],
                      "top": cands[0]["molecules"] if cands else [],
                      "candidates": _compact(cands)},
    )


_TPL_ROUTE = register_template(
    "reaction_route_target",
    "Routing ({basis}): {decision} for the target. "
    "single-step reaches stock: {reach}. complexity prior {score}/100.",
)


@mcp.tool
def reaction_route_target(
    smiles: str,
    probe: bool = True,
    top_k: int = 10,
    return_text: bool = False,
) -> dict | str:
    """Decide single-step vs multi-step retrosynthesis at a node.

    The authoritative criterion is **operational** (``probe=True``, default):
    does a single-step disconnection reach purchasable stock? It stock-checks the
    target, runs ``reaction_predict_singlestep_retro``, and checks stock
    membership of the proposed precursors:

      - ``in_stock`` — the target is already buyable (a leaf; don't expand).
      - ``single_step`` — some one-step disconnection lands *entirely* in stock;
        one call to ``reaction_predict_singlestep_retro`` solves this node.
      - ``single_then_escalate`` — a disconnection reaches stock *partially*; take
        the single step and recurse on the non-stock precursors.
      - ``multi_step_search`` — no single-step precursor is buyable; the target is
        far from stock and needs a route search (this server has no multi-step
        tool; expand the precursors node by node, or run
        ``scripts/traj_route_search.py``).

    ``chosen_disconnection`` (present for single_step / single_then_escalate) names
    the precursor set to take — its ``leaves`` are buyable (done) and its
    ``recurse_on`` precursors are the next nodes — so the caller acts without
    re-running single-step retro.

    The molecular-complexity score (SAscore/ring/fsp3/macrocycle/stereocenters)
    is returned as a **prior only** — calibration showed it does not by itself
    predict route depth (route need is a target-to-stock-gap question, not an
    absolute-complexity one), so it is advisory / a cheap fallback. When routing
    can't be probed (no stock configured or no live single-step backend), the
    tool degrades to the complexity prior and says so in ``basis``.

    Set ``probe=False`` for the cheap complexity-only prior (no model/stock calls).
    """
    if not smiles:
        return format_output(
            error_envelope("smiles is empty", ToolErrorType.INVALID_INPUT),
            _TPL_ROUTE, return_text, tool_name="reaction_route_target",
        )
    from .routing import route_node

    try:
        r = route_node(smiles, top_k=top_k, probe=probe)
    except ValueError as exc:
        return format_output(
            error_envelope(str(exc), ToolErrorType.INVALID_INPUT),
            _TPL_ROUTE, return_text, tool_name="reaction_route_target",
        )
    op, prior = r["operational"], r["complexity_prior"]
    if op is not None:
        reach = op.get("best_frac_in_stock")
        reach = "yes" if op.get("target_in_stock") or r["decision"] == "single_step" \
            else (f"{reach:.0%}" if isinstance(reach, float) else "no")
    else:
        reach = "unprobed"

    return format_output(
        r,
        _TPL_ROUTE, return_text, tool_name="reaction_route_target",
        text_payload={
            "basis": r["basis"], "decision": r["decision"], "reach": reach,
            "score": prior["score"],
        },
    )


# There is no condition-recommendation tool: the active MCP is scoped to
# retrosynthesis.


# ===========================================================================
# Bond-change extraction
# ===========================================================================

_TPL_BOND_CHANGES = register_template(
    "reaction_get_bond_changes",
    "Bond-change extraction ({mode}): {summary}",
)


@mcp.tool
def reaction_get_bond_changes(
    reaction_smiles: str | None = None,
    product_smiles: str | None = None,
    candidates: list | None = None,
    return_text: bool = False,
) -> dict | str:
    """Extract the net bond changes of a reaction via CGR (single or batch).

    This is pure bond-change *extraction* — it captures what changed, it does
    not judge plausibility. Atom-mapping is applied automatically (RXNMapper)
    when absent. No raw atom-map numbers appear in the output.

    Two input modes:
      • Single reaction — pass ``reaction_smiles`` = ``"R>>P"``.
      • Batch retro     — pass ``product_smiles`` + ``candidates`` (each a
        precursor set as ``"A.B"``, ``["A","B"]``, or ``{"molecules":[...]}``);
        each is assembled into ``precursors>>product`` and analyzed.

    Per reaction the analysis returns:
      - ``reaction_type``       : inferred class ("N-alkylation", "C-C coupling", …)
      - ``bonds_formed`` / ``bonds_broken`` : {atoms, bond_order, atom1_fg, atom2_fg, …}
      - ``bonds_order_changed`` : bonds whose order changed (oxidation/reduction)
      - ``center_atom_fg``      : FG description per reactive atom
      - ``cgr_summary`` / ``charge_changes`` / ``n_bonds_formed`` / ``n_bonds_broken``

    Batch mode wraps each candidate with ``candidate_idx`` / ``candidate_smiles``
    and returns ``{product, results, n, n_valid}``. Feed the bond changes into
    ``reaction_check_template_compatibility`` to ground them against known chemistry.
    """
    batch_mode = candidates is not None
    if batch_mode == bool(reaction_smiles):
        return format_output(
            error_envelope(
                "pass either reaction_smiles (single) OR product_smiles+candidates (batch)",
                ToolErrorType.INVALID_INPUT),
            _TPL_BOND_CHANGES, return_text, tool_name="reaction_get_bond_changes",
        )

    # ---- batch retro-candidate mode ----
    if batch_mode:
        if not product_smiles:
            return format_output(
                error_envelope("product_smiles is required in batch mode", ToolErrorType.INVALID_INPUT),
                _TPL_BOND_CHANGES, return_text, tool_name="reaction_get_bond_changes",
            )
        if not candidates:
            return format_output(
                error_envelope("candidates list is empty", ToolErrorType.INVALID_INPUT),
                _TPL_BOND_CHANGES, return_text, tool_name="reaction_get_bond_changes",
            )
        norm: list = []
        for c in candidates:
            if isinstance(c, str):
                norm.append(c)
            elif isinstance(c, (list, tuple)):
                norm.append(list(c))
            elif isinstance(c, dict):
                mols = c.get("molecules") or c.get("reactants") or []
                norm.append(mols if mols else c)
            else:
                norm.append(str(c))

        def _mapper(rxn_list):
            return _get_rxnmapper().get_attention_guided_atom_maps(rxn_list)

        try:
            results = _cgr_analyze_candidates(product_smiles, norm, rxnmapper_fn=_mapper)
        except Exception as e:
            return format_output(
                error_envelope(f"batch extraction failed: {e}", ToolErrorType.INTERNAL_ERROR),
                _TPL_BOND_CHANGES, return_text, tool_name="reaction_get_bond_changes",
            )
        n_valid = sum(1 for r in results if r.get("valid"))
        rxn_types = list(dict.fromkeys(
            r["reaction_type"] for r in results if r.get("valid") and r.get("reaction_type")
        ))
        return format_output(
            {"product": product_smiles, "results": results, "n": len(results), "n_valid": n_valid},
            _TPL_BOND_CHANGES, return_text, tool_name="reaction_get_bond_changes",
            text_payload={"mode": f"{len(results)} candidate(s)",
                          "summary": f"{n_valid}/{len(results)} valid; "
                                     f"types: {', '.join(rxn_types[:5]) or 'none'}"},
        )

    # ---- single reaction mode ----
    if ">>" not in reaction_smiles:
        return format_output(
            error_envelope("reaction_smiles must contain '>>' (R>>P)", ToolErrorType.INVALID_INPUT),
            _TPL_BOND_CHANGES, return_text, tool_name="reaction_get_bond_changes",
        )
    mapped = reaction_smiles
    mapping_confidence: float | None = None
    if ":" not in reaction_smiles:
        try:
            mapper = _get_rxnmapper()
            res = mapper.get_attention_guided_atom_maps([reaction_smiles])
            if res:
                mapped = res[0].get("mapped_rxn", reaction_smiles)
                mapping_confidence = float(res[0].get("confidence", 0.0))
        except Exception as e:
            return format_output(
                error_envelope(
                    f"RXNMapper failed: {e}. Pass an atom-mapped SMILES to bypass.",
                    ToolErrorType.UPSTREAM_ERROR,
                ),
                _TPL_BOND_CHANGES, return_text, tool_name="reaction_get_bond_changes",
            )

    result = _cgr_analyze(mapped)
    if not result.get("valid"):
        return format_output(
            error_envelope(result.get("error", "bond-change extraction failed"), ToolErrorType.INTERNAL_ERROR),
            _TPL_BOND_CHANGES, return_text, tool_name="reaction_get_bond_changes",
        )
    if mapping_confidence is not None:
        result["mapping_confidence"] = mapping_confidence

    return format_output(
        result, _TPL_BOND_CHANGES, return_text, tool_name="reaction_get_bond_changes",
        text_payload={
            "mode": "single reaction",
            "summary": f"type {result['reaction_type']}; "
                       f"{result['n_bonds_formed']} formed, {result['n_bonds_broken']} broken",
        },
    )


# ===========================================================================
# Disconnection-aware tools
# ===========================================================================

_TPL_FG = register_template(
    "reaction_detect_functional_groups",
    "Detected {n} functional group(s): {fg_names}.",
)


@mcp.tool
def reaction_detect_functional_groups(
    smiles: str,
    return_text: bool = False,
) -> dict | str:
    """Detect named functional groups in a molecule via SMARTS catalog.

    Returns a prioritized list of {fg_name, fg_type, atom_indices,
    primary_atom_idx, reactivity_note, heteroatom_adjacency}.

    fg_type categories (priority order):
      highly_electrophilic, strained_electrophile, electrophilic,
      excellent_lg, coupling_partner, nucleophilic, soft_nucleophile,
      pi_system, base_catalyst, poor_nucleophile, ewg, protecting_group, inert

    Use fg_type and atom_indices to reason about which bonds are most likely
    retrosynthetic disconnection sites, and which named reactions
    (``reaction_match_templates``) are plausible.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return format_output(
            error_envelope(f"Invalid SMILES: {smiles!r}", ToolErrorType.PARSE_ERROR),
            _TPL_FG, return_text, tool_name="reaction_detect_functional_groups",
        )
    groups = _detect_fg(smiles)
    return format_output(
        {"smiles": smiles, "functional_groups": groups, "n": len(groups)},
        _TPL_FG, return_text, tool_name="reaction_detect_functional_groups",
        text_payload={"n": len(groups), "fg_names": [g["fg_name"] for g in groups[:15]]},
    )


@mcp.tool
def reaction_in_stock(smiles: str | list[str], stock: str | None = None) -> dict:
    """Check whether molecule(s) are purchasable building blocks (in stock).

    This is the SAME leaf-termination oracle a retrosynthesis search uses: a
    molecule that is in stock is a valid leaf (stop decomposing it); one that is
    NOT in stock must be broken down further with ``reaction_predict_singlestep_retro``.
    Use it to decide, at each step, which precursors are done and which still
    need a disconnection.

    Pass a single SMILES or a list. Matching mirrors the benchmark stock exactly
    (canonical SMILES for PaRoutes, InChIKey[:14] skeleton for ZINC). The stock
    defaults to the server's REACTION_STOCK env, but an optional ``stock`` spec
    selects a specific one per call (an alias — paroutes_n1|paroutes_n5|zinc_ik14|
    emols — or ``smiles:<path>`` / ``ik14:<path>`` / ``ikfull:<path>``), so one
    server can serve several benchmarks at once. Each distinct stock is cached.

    Returns::

        {
          "stock": "<active stock label>",           # or null if unconfigured
          "results": [{"smiles", "in_stock", "valid", "match_key", ...}, ...],
          "n_total": int,
          "n_in_stock": int,
          "not_in_stock": ["<smiles>", ...],          # leaves that must be expanded
          "all_in_stock": bool                        # route is fully purchasable
        }
    """
    status = _stock_status(stock)
    if not status.get("stock_configured"):
        return {"error": status.get("error") or "no stock configured",
                "hint": status.get("hint"), "stock": None}
    items = [smiles] if isinstance(smiles, str) else list(smiles)
    results = [_stock_check_one(s, stock) for s in items]
    not_in_stock = [r["smiles"] for r in results if r.get("in_stock") is False]
    n_in = sum(1 for r in results if r.get("in_stock") is True)
    return {
        "stock": status.get("stock"),
        "results": results,
        "n_total": len(results),
        "n_in_stock": n_in,
        "not_in_stock": not_in_stock,
        "all_in_stock": len(results) > 0 and not not_in_stock,
    }


_TPL_PROPOSE = register_template(
    "reaction_propose_disconnections",
    "Proposed {n} candidate disconnection(s) (BRICS): {summary}. Candidates only — confirm with reaction_predict_singlestep_retro.",
)


@mcp.tool
def reaction_propose_disconnections(
    product_smiles: str,
    return_text: bool = False,
) -> dict | str:
    """Propose candidate retrosynthetic disconnections for a molecule via BRICS.

    BRICS ("Breaking of Retrosynthetically Interesting Chemical Substructures")
    cleaves bonds at 16 reaction-derived atom environments, so its cuts bias
    toward real disconnection sites (amide, ester, C–N, ether, biaryl, …). For
    each cleavable bond this returns:
      - ``bond``          : the atoms of the candidate disconnection
      - ``bond_type``     : rule-based label from the BRICS L-type pair
                            (e.g. "amide", "biaryl (Ar–Ar, Suzuki-type)")
      - ``synthons``      : the two fragments from cutting *only* that bond
                            (``[*]`` marks the attachment point)
      - ``synthon_sizes`` : heavy-atom counts of the two fragments (so you can
                            judge how convergent/balanced the split is)

    This tool *proposes* candidates — it does NOT verify them. BRICS is
    context-blind (it can't tell whether this specific substrate reacts) and
    never cuts fused rings, so it complements, not replaces, the learned models:
    treat each candidate as a lead and confirm feasibility with
    ``reaction_predict_singlestep_retro`` (and/or ``reaction_match_templates``). Absence of
    a BRICS cut does NOT mean the molecule is hard to make (e.g. ring-forming
    disconnections are invisible here).
    """
    if not product_smiles:
        return format_output(
            error_envelope("product_smiles is empty", ToolErrorType.INVALID_INPUT),
            _TPL_PROPOSE, return_text, tool_name="reaction_propose_disconnections",
        )
    result = _propose_disconnections(product_smiles)
    if not result.get("valid"):
        return format_output(
            error_envelope(result.get("error", "disconnection proposal failed"),
                           ToolErrorType.PARSE_ERROR),
            _TPL_PROPOSE, return_text, tool_name="reaction_propose_disconnections",
        )
    summary = "; ".join(
        f"{d['bond_type']} ({'+'.join(str(s) for s in d['synthon_sizes'])})"
        for d in result["disconnections"][:5]
    ) or "no BRICS-cleavable bond"
    return format_output(
        result, _TPL_PROPOSE, return_text, tool_name="reaction_propose_disconnections",
        text_payload={"n": result["n"], "summary": summary},
    )


_TPL_MATCH = register_template(
    "reaction_match_templates",
    "Matched {n_applicable} named reaction(s) that could produce this molecule. Top: {names}",
)


@mcp.tool
def reaction_match_templates(
    smiles: str,
    top_k: int = 10,
    name_filter: str | None = None,
    return_text: bool = False,
) -> dict | str:
    """Find which named reactions could produce a molecule (no SMARTS authoring).

    The first-layer template lookup: instead of you authoring a retro SMARTS,
    this matches ``smiles`` against a corpus of ~470 named reactions (Rxn-INSIGHT
    SMIRKS) and returns the named reactions whose product matches — ranked by how
    specifically they match. Each result carries the reaction ``name``, its
    ``template`` SMIRKS, and matched atoms; pass the ``template`` straight into
    ``reaction_check_template_compatibility`` for the detailed per-match check.

    ``name_filter`` keeps only names containing a substring (case-insensitive),
    so you can probe a specific reaction the molecule might come from. Example
    names in the corpus (partial — call the tool to see what actually applies):
      Suzuki coupling with boronic acids, Negishi coupling, Stille reaction,
      Sonogashira acetylene, Heck terminal vinyl, Buchwald-Hartwig/Ullmann
      N-arylation, Williamson Ether Synthesis, Mitsunobu aryl ether,
      Esterification of Carboxylic Acids, Schotten-Baumann to ester, Acylation of
      Nitrogen Nucleophiles by Carboxylic Acids, Acyl chloride with primary amine
      to amide, Reductive amination with aldehyde/ketone, Wittig reaction,
      Grignard from aldehyde to alcohol, Azide-nitrile click cycloaddition to
      triazole, Ugi reaction, A3 coupling, Acetal hydrolysis to aldehyde,
      Alcohol deprotection from silyl ethers.

    Returns {n_applicable, templates, names}.
    """
    result = _match_templates(smiles, top_k=top_k, name_filter=name_filter)
    if not result.get("valid"):
        return format_output(
            error_envelope(result.get("error", "template matching failed"),
                           ToolErrorType.PARSE_ERROR),
            _TPL_MATCH, return_text, tool_name="reaction_match_templates",
        )
    return format_output(
        result, _TPL_MATCH, return_text, tool_name="reaction_match_templates",
        text_payload={
            "n_applicable": result["n_applicable"],
            "names": result["names"][:top_k],
        },
    )


_TPL_TEMPLATE = register_template(
    "reaction_check_template_compatibility",
    "Template compatibility: {compatible} — {match_count} match(es), template class {template_class}.",
)


@mcp.tool
def reaction_check_template_compatibility(
    smiles: str,
    template_smarts: str,
    return_text: bool = False,
) -> dict | str:
    """Verify whether a molecule matches a specific retro SMARTS template.

    template_smarts accepts:
      - A plain SMARTS pattern (e.g. ``"[CX4][Br]"``)
      - Retro SMARTS  ``product_smarts>>reactant_smarts``  (product side matched)
      - Forward SMARTS ``reactant>>product``               (reactant side matched)

    Returns {compatible, matched_atoms, match_count, template_class, notes}.

    If you don't already have a template SMARTS, call ``reaction_match_templates``
    first — it returns real USPTO templates (with their ``retro_template`` SMARTS)
    that apply to the molecule, so you never have to author SMARTS by hand. Then
    pass a chosen ``retro_template`` here for the detailed per-match verification.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return format_output(
            error_envelope(f"Invalid SMILES: {smiles!r}", ToolErrorType.PARSE_ERROR),
            _TPL_TEMPLATE, return_text, tool_name="reaction_check_template_compatibility",
        )
    result = _check_template_compat(smiles, template_smarts)
    if "error" in result and not result.get("compatible"):
        return format_output(
            error_envelope(result["error"], ToolErrorType.INVALID_INPUT),
            _TPL_TEMPLATE, return_text, tool_name="reaction_check_template_compatibility",
        )
    return format_output(
        result, _TPL_TEMPLATE, return_text, tool_name="reaction_check_template_compatibility",
        text_payload={"compatible": result["compatible"],
                      "template_class": result.get("template_class", "unknown"),
                      "match_count": result.get("match_count", 0)},
    )


# ===========================================================================
# CGR topology tools (reaction center + graded local environment)
# ===========================================================================

_TPL_CENTER = register_template(
    "reaction_extract_reaction_center",
    "Reaction center — type: {reaction_type}. {n_center_atoms} atom(s) change bonding. "
    "Radius-0 template: {center_template}. Local environment (radius 1..{max_radius}): {local_env}.",
)


@mcp.tool
def reaction_extract_reaction_center(
    reaction_smiles: str,
    max_radius: int = 2,
    return_text: bool = False,
) -> dict | str:
    """Extract the reaction center of a reaction SMILES (R>>P) via CGRtools.

    Atom-mapping is applied automatically (RXNMapper) when not present. Goes
    beyond ``reaction_get_bond_changes`` by reporting, for every atom that
    changes, its *atom-level* rehybridization / neighbor-count / charge changes,
    plus reaction templates at graded radii: the minimal radius-0 mechanism and
    its ``local_environment`` expanded shell-by-shell (radius 1..``max_radius``).

    ``max_radius`` (default 2, capped at 3) controls how many bond-shells of
    steric/electronic context around the center are returned — radius-0 alone is
    the bare mechanism, but the surrounding shells are what decide whether a
    template actually applies (an ester α to a ring behaves differently from an
    isolated one).

    Returns:
      - ``reaction_type``      : inferred class
      - ``center_atoms``       : per-atom {element, fg, hybridization,
                                  hybridization_change, neighbor_change,
                                  charge_change, in_ring}
      - ``center_bonds``       : {atoms, change (formed/broken/order_changed), order}
      - ``center_template``    : radius-0 CGR signature — the bare mechanism
      - ``local_environment``  : [{radius, n_atoms, template}] for radius 1..max_radius
      - ``n_components``       : number of disconnected reaction-center fragments

    Use to pinpoint exactly which atoms/bonds react and how their bonding
    environment changes — e.g. to validate a mechanism or seed a graded template.
    """
    if ">>" not in reaction_smiles:
        return format_output(
            error_envelope("reaction_smiles must contain '>>' (R>>P)", ToolErrorType.INVALID_INPUT),
            _TPL_CENTER, return_text, tool_name="reaction_extract_reaction_center",
        )
    mapped, conf, err = _ensure_mapped(reaction_smiles)
    if err:
        return format_output(
            error_envelope(err, ToolErrorType.UPSTREAM_ERROR),
            _TPL_CENTER, return_text, tool_name="reaction_extract_reaction_center",
        )
    result = _cgr_extract_center(mapped, max_radius=max_radius)
    if not result.get("valid"):
        return format_output(
            error_envelope(result.get("error", "reaction-center extraction failed"), ToolErrorType.INTERNAL_ERROR),
            _TPL_CENTER, return_text, tool_name="reaction_extract_reaction_center",
        )
    if conf is not None:
        result["mapping_confidence"] = conf
    local_env = result.get("local_environment", [])
    local_env_text = "; ".join(
        f"r{sh['radius']}={sh['template']}" for sh in local_env
    ) or "none"
    return format_output(
        result, _TPL_CENTER, return_text, tool_name="reaction_extract_reaction_center",
        text_payload={
            "reaction_type": result["reaction_type"],
            "n_center_atoms": len(result["center_atoms"]),
            "center_template": result["center_template"],
            "max_radius": local_env[-1]["radius"] if local_env else 0,
            "local_env": local_env_text,
        },
    )


_TPL_REACTION_TEMPLATE = register_template(
    "reaction_extract_reaction_template",
    "Extracted retro template (rdchiral): {retro_template}",
)


@mcp.tool
def reaction_extract_reaction_template(
    reaction_smiles: str,
    return_text: bool = False,
) -> dict | str:
    """Extract a reaction-specific retro template (SMARTS) from a reaction via rdchiral.

    Atom-mapping is applied automatically (RXNMapper) when absent. Where
    ``reaction_match_templates`` matches a *molecule* against a fixed corpus of
    ~470 named reactions, this derives a *fresh* generalized retro template from
    the given reaction itself — so it covers transformations outside the named
    set. rdchiral abstracts the changing atoms (leaving/attacking groups, H
    counts, degree, charge) into a reusable ``product>>reactants`` SMARTS.

    Returns:
      - ``retro_template``     : rdchiral ``reaction_smarts`` (product>>reactants),
                                 feedable straight into
                                 ``reaction_check_template_compatibility``
      - ``necessary_reagent``  : reagent context rdchiral judged essential (or null)
      - ``intra_only``         : template only valid intramolecularly
      - ``dimer_only``         : template only valid as a dimerization

    Pairs with ``reaction_extract_reaction_center`` (mechanism/atom view) and
    ``reaction_check_template_compatibility`` (apply the template to a molecule).
    """
    if ">>" not in reaction_smiles:
        return format_output(
            error_envelope("reaction_smiles must contain '>>' (R>>P)", ToolErrorType.INVALID_INPUT),
            _TPL_REACTION_TEMPLATE, return_text, tool_name="reaction_extract_reaction_template",
        )
    mapped, conf, err = _ensure_mapped(reaction_smiles)
    if err:
        return format_output(
            error_envelope(err, ToolErrorType.UPSTREAM_ERROR),
            _TPL_REACTION_TEMPLATE, return_text, tool_name="reaction_extract_reaction_template",
        )
    result = _extract_reaction_template(mapped)
    if not result.get("valid"):
        return format_output(
            error_envelope(result.get("error", "template extraction failed"), ToolErrorType.INTERNAL_ERROR),
            _TPL_REACTION_TEMPLATE, return_text, tool_name="reaction_extract_reaction_template",
        )
    if conf is not None:
        result["mapping_confidence"] = conf
    return format_output(
        result, _TPL_REACTION_TEMPLATE, return_text, tool_name="reaction_extract_reaction_template",
        text_payload={"retro_template": result["retro_template"]},
    )


def warmup() -> None:
    """Pre-load in-process models so the first real request is not a cold hit.

    Loads RXNMapper and the template corpus, and probes backend
    health. Disable with ``REACTION_WARMUP=0``.
    """
    import os as _os
    if _os.getenv("REACTION_WARMUP", "1").strip().lower() in {"0", "false", "no", "off"}:
        print("[reaction-warmup] skipped (REACTION_WARMUP=0)", flush=True)
        return
    pool = get_pool()
    try:
        _get_rxnmapper()
    except Exception:
        pass
    try:
        from .templates import library_status  # compile the ~42.5k-template corpus once
        library_status()
    except Exception:
        pass
    try:
        pool.health()
    except Exception:
        pass
    print("[reaction-warmup] done", flush=True)
