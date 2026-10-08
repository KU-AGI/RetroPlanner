"""Core contracts for the reaction model pool.

A :class:`Task` names what a model does. A :class:`Candidate` is one normalized
proposal from one backend (a product, a precursor set, or a condition
component). A :class:`Predictor` is a uniform wrapper over a model: it says
which tasks it serves, whether it is currently reachable, and how to run one.

The request shape per task (the ``request`` dict passed to ``Predictor.predict``):

  - FORWARD:    {"reactants": [smi, ...], "reagents": [smi, ...], "top_k": int}
                (no backend is configured; kept so forward round-trip verification
                 can be re-enabled by adding a forward BackendSpec)
  - RETRO:      {"product": smi, "top_k": int}

A predictor returns a list of :class:`Candidate`. The pool wraps that in a
:class:`BackendResult` (adding ok/error/elapsed) so partial failures never sink
the whole request.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Task(str, Enum):
    FORWARD = "forward"  # no backend configured; see config.py
    RETRO = "retro"  # single-step

    @classmethod
    def from_str(cls, value: str) -> "Task":
        try:
            return cls(value.strip().lower())
        except ValueError as exc:
            raise ValueError(
                f"unknown task {value!r}; expected one of {[t.value for t in cls]}"
            ) from exc


@dataclass
class Candidate:
    """One normalized proposal from one backend.

    ``molecules`` holds the canonical SMILES that make up this candidate:
      - forward  -> [product] (occasionally several products)
      - retro    -> the precursor set of one disconnection (>=1 SMILES)
      - condition-> [component] (one catalyst/solvent/reagent SMILES or name)

    ``key`` is the order-independent identity used for cross-backend voting.
    """

    molecules: list[str]
    backend: str
    rank: int | None = None
    score: float | None = None  # backend-native confidence/score (higher = better)
    role: str | None = None  # condition role: catalyst|solvent|reagent
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        base = ".".join(sorted(m for m in self.molecules if m))
        return f"{self.role}:{base}" if self.role else base

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "molecules": self.molecules,
            "backend": self.backend,
            "rank": self.rank,
            "score": self.score,
        }
        if self.role is not None:
            d["role"] = self.role
        if self.raw:
            d["raw"] = self.raw
        return d


@dataclass
class BackendResult:
    """Outcome of asking one backend for one task."""

    backend: str
    task: Task
    ok: bool
    candidates: list[Candidate] = field(default_factory=list)
    error: str | None = None
    error_type: str | None = None
    elapsed_ms: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "task": self.task.value,
            "ok": self.ok,
            "n_candidates": len(self.candidates),
            "error": self.error,
            "error_type": self.error_type,
            "elapsed_ms": self.elapsed_ms,
            "meta": self.meta,
        }


class Predictor(abc.ABC):
    """Uniform wrapper over one reaction model.

    Subclasses set ``name`` and ``tasks`` and implement :meth:`predict` and
    :meth:`available`. Implementations must be defensive: ``available`` should
    never raise, and ``predict`` should return ``[]`` (or raise a plain
    ``Exception`` the pool will catch) rather than crash the server.
    """

    name: str = "predictor"
    tasks: frozenset[Task] = frozenset()
    # Free-form description surfaced by reaction_list_models.
    description: str = ""

    def supports(self, task: Task) -> bool:
        return task in self.tasks

    def available(self) -> tuple[bool, str | None]:
        """Return (is_reachable, reason_if_not). Must not raise."""
        return True, None

    @abc.abstractmethod
    def predict(self, task: Task, request: dict[str, Any]) -> list[Candidate]:
        """Run ``task`` and return candidates. May raise on real failure."""
        raise NotImplementedError


__all__ = ["Task", "Candidate", "BackendResult", "Predictor"]
