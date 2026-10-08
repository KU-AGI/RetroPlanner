"""ModelPool: build predictors from config, probe health, fan out a request.

The pool never lets one backend sink the request: every :meth:`run` call returns
one :class:`BackendResult` per attempted backend, with failures captured as
``ok=False`` rather than raised. Backend objects are cheap to construct (no model
load at import), so the pool is a process-wide singleton built lazily.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .base import BackendResult, Predictor, Task
from .config import BACKENDS, BackendSpec
from .transport import TransportError


def _build(spec: BackendSpec) -> Predictor:
    if spec.kind == "http":
        from .backends.http_backend import HttpBackend

        return HttpBackend(spec)
    raise ValueError(f"unknown backend kind: {spec.kind!r}")


class ModelPool:
    def __init__(self, specs: list[BackendSpec] | None = None) -> None:
        self._specs = specs if specs is not None else BACKENDS
        self._predictors: dict[str, Predictor] = {}
        for spec in self._specs:
            try:
                self._predictors[spec.name] = _build(spec)
            except Exception:  # noqa: BLE001 - a broken spec must not kill the pool
                continue

    # ------------------------------------------------------------- selection
    def _spec(self, name: str) -> BackendSpec | None:
        return next((s for s in self._specs if s.name == name), None)

    def predictors_for(
        self, task: Task, models: list[str] | None = None
    ) -> list[Predictor]:
        wanted = set(models) if models else None
        out: list[Predictor] = []
        for spec in self._specs:
            if task not in spec.tasks or not spec.enabled():
                continue
            if wanted is not None and spec.name not in wanted:
                continue
            p = self._predictors.get(spec.name)
            if p is not None:
                out.append(p)
        return out

    # ---------------------------------------------------------------- health
    def health(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for spec in self._specs:
            p = self._predictors.get(spec.name)
            if p is None:
                rows.append({"backend": spec.name, "live": False,
                             "reason": "failed to construct"})
                continue
            try:
                live, reason = p.available()
            except Exception as exc:  # noqa: BLE001
                live, reason = False, f"available() raised: {exc}"
            rows.append(
                {
                    "backend": spec.name,
                    "kind": spec.kind,
                    "tasks": [t.value for t in spec.tasks],
                    "enabled": spec.enabled(),
                    "live": bool(live),
                    "reason": reason,
                    "url": spec.url() if spec.kind == "http" else None,
                    "description": spec.description,
                }
            )
        return rows

    # --------------------------------------------------------------- dispatch
    def run(
        self,
        task: Task,
        request: dict[str, Any],
        models: list[str] | None = None,
        max_workers: int = 8,
    ) -> list[BackendResult]:
        predictors = self.predictors_for(task, models)

        def _call(p: Predictor) -> BackendResult:
            t0 = time.perf_counter()
            try:
                cands = p.predict(task, request)
                return BackendResult(
                    backend=p.name,
                    task=task,
                    ok=True,
                    candidates=cands,
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                )
            except TransportError as exc:
                return BackendResult(
                    backend=p.name, task=task, ok=False, error=str(exc),
                    error_type="upstream_error",
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                )
            except (ValueError, NotImplementedError) as exc:
                return BackendResult(
                    backend=p.name, task=task, ok=False, error=str(exc),
                    error_type="invalid_input"
                    if isinstance(exc, ValueError)
                    else "unsupported",
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                )
            except Exception as exc:  # noqa: BLE001
                return BackendResult(
                    backend=p.name, task=task, ok=False, error=str(exc),
                    error_type="internal_error",
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                )

        if not predictors:
            return []
        if len(predictors) == 1:
            return [_call(predictors[0])]
        with ThreadPoolExecutor(max_workers=min(max_workers, len(predictors))) as ex:
            return list(ex.map(_call, predictors))


_POOL: ModelPool | None = None


def get_pool() -> ModelPool:
    global _POOL
    if _POOL is None:
        _POOL = ModelPool()
    return _POOL


__all__ = ["ModelPool", "get_pool"]
