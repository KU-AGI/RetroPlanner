"""Consensus over heterogeneous backends -- the "which molecule reacts" view.

Two complementary signals:

* :func:`aggregate` -- cross-model **voting**. Canonicalize every candidate to an
  order-independent key, group across backends, and rank by how many independent
  models proposed it (agreement), breaking ties by mean reciprocal within-backend
  rank. Native scores live on different scales (priors, beam log-probs, template
  counts), so they are kept per-backend for transparency, not summed.

* :func:`round_trip_filter` -- **round-trip consistency** (Schwaller et al.). A
  retro disconnection is only real if running the forward pool on its precursors
  regenerates the target product. No forward backend is configured in this repo
  (config.py), so the filter currently annotates each candidate as unverified
  and leaves the order alone; adding a forward BackendSpec re-enables it.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .base import BackendResult, Task
from .smiles import canonical, retro_conserves_mass

# Reciprocal-rank-fusion smoothing constant (Cormack et al. 2009).
RRF_K0 = 60

# Tie-break / single-model preference, R-SMILES first, then LocalRetro. Used by
# backbone="single" to choose the model when
# weights are equal, and as a deterministic tie-break elsewhere.
_MODEL_PRIORITY = ("rsmiles", "localretro")

# Candidate-selection backbones.
SELECTION_BACKBONES = ("consensus", "single", "plurality", "rrf", "union")


def _prio(backend: str) -> int:
    return _MODEL_PRIORITY.index(backend) if backend in _MODEL_PRIORITY \
        else len(_MODEL_PRIORITY)


def aggregate(
    results: list[BackendResult],
    top_k: int = 10,
    *,
    weights: dict[str, float] | None = None,
    backbone: str = "consensus",
    gate_tau: int = 0,
    single_model: str | None = None,
    product: str | None = None,
    mass_guard: bool = True,
) -> dict[str, Any]:
    """Fold per-backend candidates into one ranked list under a selection backbone.

    ``backbone`` selects HOW the per-model candidate lists are combined:

    * ``consensus`` -- approval vote: rank by (#backends proposing) + mean
      reciprocal within-backend rank. The historical default.
    * ``single``    -- SOTA: use only ONE model's candidate list; no fusion.
      ``single_model`` pins which model (fail-closed: if that model did not
      respond, ABSTAIN instead of silently downgrading to a weaker live model).
      ``single_model="auto"`` or None restores legacy best-available behaviour:
      the highest-weighted live model, ties broken by :data:`_MODEL_PRIORITY`.
    * ``plurality`` -- each model casts ONE vote, for its top-1 only; most votes
      wins (RRF tie-break).
    * ``rrf``       -- reciprocal-rank fusion: every candidate at every rank
      contributes ``w / (RRF_K0 + rank)``.
    * ``union``     -- all candidates pooled (pass@k upper bound), RRF-ordered.

    ``gate_tau`` (>0) enables agreement gating: if fewer than ``gate_tau`` models
    agree on their top-1 pick, the result is flagged ``abstained`` (the caller
    decides whether to answer). Native per-model scores stay per-backend.

    ``mass_guard`` (needs ``product``) drops retro candidates whose precursors cannot supply
    the product's skeleton -- see :func:`retro_conserves_mass`. It runs BEFORE voting, so a
    degenerate proposal cannot earn agreement, and the count of what it removed is returned as
    ``mass_rejected``. Without ``product`` it is a no-op, which keeps forward/condition tasks
    and any caller that does not pass a product unaffected.
    """
    weights = weights or {}
    backbone = (backbone or "consensus").lower()
    if backbone not in SELECTION_BACKBONES:
        backbone = "consensus"
    ok = [r for r in results if r.ok]
    failed = [
        {"backend": r.backend, "error": r.error, "error_type": r.error_type}
        for r in results
        if not r.ok
    ]
    n_ok = len(ok)
    ok_backends = [r.backend for r in ok]

    groups: dict[str, dict[str, Any]] = {}
    mass_rejected = 0
    for r in ok:
        for c in r.candidates:
            key = c.key
            if not key:
                continue
            if mass_guard and product and not retro_conserves_mass(product, c.molecules):
                mass_rejected += 1
                continue
            g = groups.setdefault(
                key,
                {"molecules": c.molecules, "role": c.role, "per_backend": {}},
            )
            rank = c.rank if isinstance(c.rank, int) else 0
            prev = g["per_backend"].get(c.backend)
            # keep this backend's best (lowest-rank) proposal of this candidate
            if prev is None or rank < prev["rank"]:
                g["per_backend"][c.backend] = {
                    "backend": c.backend,
                    "rank": rank,
                    "score": c.score,
                }

    rows: list[dict[str, Any]] = []
    for key, g in groups.items():
        per = list(g["per_backend"].values())
        votes = sum(weights.get(p["backend"], 1.0) for p in per)
        mrr = sum(1.0 / (p["rank"] + 1) for p in per) / len(per)
        best_rank = min(p["rank"] for p in per)
        rrf_score = sum(
            weights.get(p["backend"], 1.0) / (RRF_K0 + p["rank"]) for p in per
        )
        # plurality: weighted count of backends that ranked this candidate #1
        plurality_votes = sum(
            weights.get(p["backend"], 1.0) for p in per if p["rank"] == 0
        )
        rows.append(
            {
                "molecules": g["molecules"],
                "key": key,
                "role": g["role"],
                "votes": round(votes, 4),
                "n_backends": len(per),
                "agreement": round(len(per) / n_ok, 4) if n_ok else 0.0,
                "best_rank": best_rank,
                "mean_reciprocal_rank": round(mrr, 4),
                "consensus_score": round(votes + mrr, 6),
                "rrf_score": round(rrf_score, 6),
                "plurality_votes": round(plurality_votes, 4),
                "backends": sorted(p["backend"] for p in per),
                "per_backend": sorted(per, key=lambda p: p["rank"]),
            }
        )

    # ---- agreement gate: how many models agree on their (shared) top-1 pick ----
    top1_counts: dict[str, int] = {}
    for r in ok:
        for c in r.candidates:
            rank = c.rank if isinstance(c.rank, int) else 0
            if rank == 0 and c.key:
                top1_counts[c.key] = top1_counts.get(c.key, 0) + 1
                break
    agreement = max(top1_counts.values()) if top1_counts else 0
    abstained = bool(gate_tau and agreement < gate_tau)

    # ---- rank rows under the chosen backbone --------------------------------
    required_model = None
    effective_model = None
    if backbone == "single":
        pin = (single_model or "").strip().lower()
        if pin and pin != "auto":
            # fail-closed: the pinned model MUST have responded, else abstain
            # rather than silently answering with a weaker live model.
            required_model = pin
            chosen = pin if pin in ok_backends else None
            if chosen is None:
                abstained = True
        elif ok_backends:
            # legacy best-available: highest-weighted live model, _prio tie-break
            chosen = min(ok_backends, key=lambda b: (-weights.get(b, 1.0), _prio(b)))
        else:
            chosen = None
        effective_model = chosen
        if chosen is None:
            rows = []
        else:
            rows = [r for r in rows if chosen in r["backends"]]
            rows.sort(
                key=lambda d: next(
                    p["rank"] for p in d["per_backend"] if p["backend"] == chosen
                )
            )
    elif backbone == "plurality":
        rows.sort(key=lambda d: (d["plurality_votes"], d["rrf_score"]), reverse=True)
    elif backbone in ("rrf", "union"):
        rows.sort(key=lambda d: (d["rrf_score"], -d["best_rank"]), reverse=True)
    else:  # consensus
        rows.sort(key=lambda d: (d["consensus_score"], -d["best_rank"]), reverse=True)

    return {
        "candidates": rows[: max(1, top_k)],
        "n_backends": n_ok,
        "backends_ok": sorted(r.backend for r in ok),
        "backends_failed": failed,
        "selection": backbone,
        "single_required": required_model,
        "single_effective": effective_model,
        "agreement": agreement,
        "mass_rejected": mass_rejected,
        "gate_tau": gate_tau,
        "abstained": abstained,
    }


def round_trip_filter(
    pool,
    candidates: list[dict[str, Any]],
    product_smiles: str,
    *,
    forward_models: list[str] | None = None,
    forward_top_k: int = 5,
    rerank: bool = True,
    mode: str = "soft",
) -> list[dict[str, Any]]:
    """Annotate retro candidates with forward round-trip consistency.

    For each candidate (precursor set), run the forward pool and check whether
    the target product appears in the predicted products. Adds a ``round_trip``
    block per candidate. ``mode`` controls how the verification is applied:

    * ``soft`` (default) -- round-trip-confirmed candidates are floated to the
      top, preserving the incoming (backbone) order within each tier.
    * ``hard`` -- drop candidates that do NOT round-trip (keep the original list
      only if NONE round-trips, so the tool never returns empty).
    * ``off`` -- annotate only, no reordering or filtering.

    With no enabled forward backend (the default here), every candidate gets
    ``round_trip = {"matched": None, ...}`` and the list is returned unchanged.
    """
    mode = (mode or "soft").lower()
    target = canonical(product_smiles)

    # No forward backend configured/enabled -> nothing can verify. Mark every
    # candidate unverified (matched=None, distinct from a failed round trip) and
    # return the list untouched, so neither soft nor hard mode reorders or drops.
    if not pool.predictors_for(Task.FORWARD, forward_models):
        for cand in candidates:
            cand["round_trip"] = {"matched": None, "rank": None, "predicted": [],
                                  "forward_backends": [],
                                  "reason": "no forward backend configured"}
        return candidates

    # One forward run per UNIQUE precursor set, executed concurrently -- the
    # forward backends are warm services / GPU, so N candidates verify in roughly
    # the wall-time of one instead of N. (Pass forward_models to restrict which backends
    # verify.)
    uniq: dict[str, list[str]] = {}
    for cand in candidates:
        mols = cand.get("molecules") or []
        uniq.setdefault(".".join(sorted(mols)), mols)

    def _verify(mols: list[str]) -> dict[str, Any]:
        fwd = pool.run(
            Task.FORWARD,
            {"reactants": mols, "reagents": [], "top_k": forward_top_k},
            models=forward_models,
        )
        agg = aggregate(fwd, top_k=forward_top_k)
        predicted = [c["molecules"][0] for c in agg["candidates"] if c["molecules"]]
        matched_rank = next(
            (i for i, p in enumerate(predicted)
             if target is not None and canonical(p) == target),
            None,
        )
        return {
            "matched": matched_rank is not None,
            "rank": matched_rank,
            "predicted": predicted,
            "forward_backends": agg["backends_ok"],
        }

    keys = list(uniq)
    if len(keys) <= 1:
        cache = {k: _verify(uniq[k]) for k in keys}
    else:
        with ThreadPoolExecutor(max_workers=min(8, len(keys))) as ex:
            cache = dict(zip(keys, ex.map(lambda k: _verify(uniq[k]), keys)))

    for cand in candidates:
        cand["round_trip"] = cache[".".join(sorted(cand.get("molecules") or []))]

    if mode == "hard":
        kept = [c for c in candidates if c.get("round_trip", {}).get("matched")]
        return kept or candidates  # never return empty: fall back to all
    if mode == "soft" and rerank:
        # Stable sort: float round-trip-confirmed to the top while preserving the
        # incoming backbone order within the matched / unmatched tiers.
        candidates.sort(
            key=lambda c: 0 if c.get("round_trip", {}).get("matched") else 1
        )
    return candidates


__all__ = ["aggregate", "round_trip_filter"]
