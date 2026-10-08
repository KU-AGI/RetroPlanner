"""Generic warm-HTTP backend.

Wraps any model served over the standard JSON contract (see config.py). One
adapter instance per :class:`~reaction_mcp.pool.config.BackendSpec`; the request
payload is built per task and the response is normalized into
:class:`~reaction_mcp.pool.base.Candidate` objects, accepting both the canonical
``candidates`` shape and the legacy ``predictions`` / ``precursors`` shapes.
"""
from __future__ import annotations

from typing import Any

from ..base import Candidate, Predictor, Task
from ..config import BackendSpec
from ..smiles import canonical, canonical_set
from ..transport import TransportError, health, post_json


def build_request(task: Task, request: dict[str, Any]) -> dict[str, Any]:
    """Translate a pool request into the on-the-wire payload."""
    if task is Task.FORWARD:
        return {
            "task": "forward",
            "reactants_smiles": request.get("reactants", []),
            "reagents_smiles": request.get("reagents", []) or [],
            "top_k": int(request.get("top_k", 5)),
        }
    if task is Task.RETRO:
        return {
            "task": task.value,
            "product_smiles": request.get("product") or request.get("target"),
            "top_k": int(request.get("top_k", 10)),
        }
    raise NotImplementedError(task)


def _to_molecules(d: dict) -> list[str]:
    """Pull the SMILES making up a candidate from any known field shape."""
    for key in ("molecules", "reactants", "precursors"):
        val = d.get(key)
        if isinstance(val, list) and val:
            return canonical_set([str(x) for x in val])
        if isinstance(val, str) and val:
            return canonical_set(val.split("."))
    for key in ("smiles", "product", "component"):
        val = d.get(key)
        if isinstance(val, str) and val:
            c = canonical(val)
            return [c] if c else [val]
    return []


def normalize_response(resp: dict, backend: str) -> list[Candidate]:
    if not isinstance(resp, dict):
        raise TransportError(f"{backend}: response was not a JSON object")
    if resp.get("error"):
        raise RuntimeError(resp["error"])
    rows = resp.get("candidates")
    if rows is None:  # legacy shapes
        rows = resp.get("predictions") or resp.get("precursors") or []
    out: list[Candidate] = []
    for i, d in enumerate(rows):
        if not isinstance(d, dict):
            continue
        mols = _to_molecules(d)
        if not mols:
            continue
        score = d.get("score")
        if score is None:
            score = d.get("confidence")
        out.append(
            Candidate(
                molecules=mols,
                backend=backend,
                rank=d.get("rank", i),
                score=float(score) if isinstance(score, (int, float)) else None,
                role=d.get("role"),
                raw={
                    k: v
                    for k, v in d.items()
                    if k not in {"molecules", "smiles", "score", "rank", "role"}
                },
            )
        )
    return out


class HttpBackend(Predictor):
    def __init__(self, spec: BackendSpec) -> None:
        self.spec = spec
        self.name = spec.name
        self.tasks = frozenset(spec.tasks)
        self.description = spec.description

    def available(self) -> tuple[bool, str | None]:
        url = self.spec.url()
        if not url:
            return False, "no service URL configured"
        return health(url)

    def predict(self, task: Task, request: dict[str, Any]) -> list[Candidate]:
        url = self.spec.url()
        if not url:
            raise RuntimeError(f"{self.name}: no service URL configured")
        payload = build_request(task, request)
        resp = post_json(url, payload, self.spec.timeout_s())
        return normalize_response(resp, self.name)


__all__ = ["HttpBackend", "build_request", "normalize_response"]
