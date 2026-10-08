"""Declarative backend registry.

One :class:`BackendSpec` per model. The pool carries two single-step retro models,
R-SMILES (``rsmiles``) and LocalRetro (``localretro``), both warm HTTP servers. Only
R-SMILES is on by default; LocalRetro is opt-in (``REACTION_ENABLE_LOCALRETRO=1``). No forward backend is configured, so :data:`Task.FORWARD` has
no backend and forward round-trip verification is a no-op (see
:func:`~reaction_mcp.pool.consensus.round_trip_filter`).

The :class:`~reaction_mcp.pool.registry.ModelPool` reads this list, instantiates a predictor per spec, and probes health. A backend
is *configured* here but only *live* if its warm server answers ``/health`` (or,
for in-process backends, its package imports). Override any service URL / timeout
/ enablement via the listed environment variables, so deployment never requires
editing code.

Wire contract for warm servers (``scripts/*_server.py``): POST a task request,
get back ``{"candidates": [{"molecules": [...], "score": float, "rank": int,
"role": str?, "raw": {...}}, ...]}``. The HTTP adapter also accepts the legacy
``predictions`` / ``precursors`` shapes for back-compat.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .base import Task


@dataclass(frozen=True)
class BackendSpec:
    name: str
    kind: str  # "http"
    tasks: tuple[Task, ...]
    description: str
    enabled_default: bool = True
    url_env: str | None = None
    default_url: str | None = None
    timeout_env: str | None = None
    default_timeout_s: float = 120.0
    enable_env: str | None = None  # truthy/falsy override of enabled_default
    options: dict = field(default_factory=dict)

    # --- resolved-at-call helpers (read env each time; cheap, deploy-friendly) ---
    def url(self) -> str | None:
        if self.url_env and (v := os.getenv(self.url_env)):
            return v
        return self.default_url

    def timeout_s(self) -> float:
        if self.timeout_env and (v := os.getenv(self.timeout_env)):
            try:
                return float(v)
            except ValueError:
                pass
        return self.default_timeout_s

    def enabled(self) -> bool:
        if self.enable_env and (v := os.getenv(self.enable_env)) is not None:
            return v.strip().lower() not in {"0", "false", "no", "off", ""}
        return self.enabled_default


# Default warm-server ports. Each server is launched in its own conda env; see
# scripts/<name>_server.py, scripts/launch_replicas.sh and docs/README.md "Model pool".
_HOST = os.getenv("REACTION_POOL_HOST", "127.0.0.1")


def _u(port: int) -> str:
    return f"http://{_HOST}:{port}/predict"


BACKENDS: list[BackendSpec] = [
    # --- warm HTTP single-step retro servers (each in its own env) ---
    BackendSpec(
        name="rsmiles",
        kind="http",
        tasks=(Task.RETRO,),
        description=(
            "R-SMILES (Zhong et al. 2022): augmented SMILES seq2seq retrosynthesis. "
            "GitHub: https://github.com/otori-bird/retrosynthesis. "
            "Warm server: scripts/rsmiles_server.py port 8100. "
            "Configure: REACTION_RSMILES_URL, REACTION_RSMILES_CHECKPOINT."
        ),
        url_env="REACTION_RSMILES_URL",
        default_url=_u(8100),
        timeout_env="REACTION_RETRO_TIMEOUT_S",
        enabled_default=True,
        enable_env="REACTION_ENABLE_RSMILES",
    ),
    BackendSpec(
        name="localretro",
        kind="http",
        tasks=(Task.RETRO,),
        description=(
            "LocalRetro (Chen et al. 2021): template-based local retrosynthesis via "
            "GNN atom/bond edit scoring. "
            "Warm server: scripts/localretro_server.py port 8097. "
            "Configure: REACTION_LOCALRETRO_URL, REACTION_OPENRETRO_REPO."
        ),
        url_env="REACTION_LOCALRETRO_URL",
        default_url=_u(8097),
        timeout_env="REACTION_RETRO_TIMEOUT_S",
        enabled_default=False,          # opt-in: R-SMILES is the default setup
        enable_env="REACTION_ENABLE_LOCALRETRO",
    ),
]

def specs_for(task: Task) -> list[BackendSpec]:
    return [s for s in BACKENDS if task in s.tasks and s.enabled()]


def spec_by_name(name: str) -> BackendSpec | None:
    for s in BACKENDS:
        if s.name == name:
            return s
    return None


__all__ = ["BackendSpec", "BACKENDS", "specs_for", "spec_by_name"]
