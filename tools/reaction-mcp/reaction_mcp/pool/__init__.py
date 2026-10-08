"""reaction-mcp model pool.

A task-oriented ensemble layer over heterogeneous reaction models. Each model
is wrapped as a :class:`~reaction_mcp.pool.base.Predictor` that advertises which
:class:`~reaction_mcp.pool.base.Task` it can serve (single-step retro; forward
has no backend). The configured models are R-SMILES and LocalRetro (see
``config.py``). The
:class:`~reaction_mcp.pool.registry.ModelPool` discovers
the configured predictors, probes their health, fans a request out to every
live backend that supports the task, and the :mod:`consensus` layer aggregates
the per-backend candidates into a single ranked list -- the "which molecule
actually reacts" view.

Backends live in different conda envs, so the retro models run as warm HTTP
microservices (``scripts/rsmiles_server.py``, ``scripts/localretro_server.py``); the pool itself is a thin client
plus the consensus aggregator and stays importable in the light reaction-mcp
env even when no backend is up.
"""
from __future__ import annotations

from .base import (
    Candidate,
    BackendResult,
    Predictor,
    Task,
)
from .consensus import aggregate, round_trip_filter
from .registry import ModelPool, get_pool

__all__ = [
    "Candidate",
    "BackendResult",
    "Predictor",
    "Task",
    "aggregate",
    "round_trip_filter",
    "ModelPool",
    "get_pool",
]
