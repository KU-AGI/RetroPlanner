"""Tiny stdlib HTTP client for warm backend microservices.

Every warm server (``scripts/*_server.py``) speaks the same JSON contract:

    POST {any path}  in: <task request>   out: <task response>
    GET  /health     -> {"ok": true, ...}

Kept dependency-free (urllib) so the reaction-mcp env needs nothing extra.
"""
from __future__ import annotations

import json
from urllib import error as urlerror
from urllib import request as urlrequest


class TransportError(RuntimeError):
    """Raised when a backend HTTP call fails (connect/timeout/HTTP/decode)."""


def post_json(url: str, payload: dict, timeout_s: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8")
    except urlerror.URLError as exc:  # connection refused, DNS, timeout, HTTP
        raise TransportError(f"{url}: {getattr(exc, 'reason', exc)}") from exc
    except OSError as exc:  # socket timeout etc.
        raise TransportError(f"{url}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise TransportError(f"{url}: non-JSON response: {exc}") from exc


def health(url: str, timeout_s: float = 2.0) -> tuple[bool, str | None]:
    """GET ``{base}/health``; return (ok, reason_if_down). Never raises."""
    base = url.rstrip("/")
    # Strip a trailing path segment so /predict -> /health works too.
    if "/" in base.split("://", 1)[-1]:
        base = base.rsplit("/", 1)[0]
    health_url = f"{base}/health"
    try:
        with urlrequest.urlopen(health_url, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8") or "{}")
        if body.get("ok", True):
            return True, None
        return False, str(body)
    except Exception as exc:  # noqa: BLE001 - health must never raise
        return False, f"unreachable: {exc}"


__all__ = ["TransportError", "post_json", "health"]
