#!/usr/bin/env python
"""Reusable warm-server harness for reaction-pool backends.

Each model backend runs as a persistent HTTP service in ITS OWN conda env, so
the light reaction-mcp env stays free of conflicting deps. They all speak one
contract (matched by reaction_mcp/pool -- HttpBackend):

    POST {any path}
      in : {"task": "forward"|"retro"|"condition", ...}
           forward:   {"reactants_smiles": [...], "reagents_smiles": [...], "top_k": int}
           retro:     {"product_smiles": str, "top_k": int}
           condition: {"reaction_smiles": str?, "rxn_class": str?,
                       "targets": [...], "top_k": int}
      out: {"candidates": [{"molecules": [...], "score": float, "rank": int,
                            "role": str?, "raw": {...}}, ...]}
           or {"error": str, "error_type": str}
    GET /health -> {"ok": true, "backend": str}

A concrete server imports :func:`serve` and passes a ``predict(task, request)``
callable that returns a list of candidate dicts. The harness serializes calls
(``--serialize``, default on) because most ML models are not thread-safe.
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler
from typing import Callable, Dict, List, Optional

# ThreadingHTTPServer is py3.7+. Older backend envs may pin python 3.6, so fall
# back to the equivalent (ThreadingMixIn + HTTPServer) there. No `from __future__
# import annotations` for the same reason -- keep every annotation py3.6-safe
# (use typing.Optional, not the py3.10+ `X | None`).
try:
    from http.server import ThreadingHTTPServer
except ImportError:  # python 3.6
    import socketserver
    from http.server import HTTPServer

    class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
        daemon_threads = True

# List[Dict] (not list[dict]) so this harness also imports under py3.7 backends.
PredictFn = Callable[[str, Dict], List[Dict]]


def serve(
    predict: PredictFn,
    *,
    backend: str,
    default_port: int,
    serialize: bool = True,
    warmup: Optional[Callable[[], None]] = None,
) -> None:
    ap = argparse.ArgumentParser(description=f"{backend} warm server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=default_port)
    args = ap.parse_args()

    lock = threading.Lock() if serialize else None

    # Readiness flag. A backend is "ready" only once warmup (= model load for
    # lazy servers) has succeeded. /health is a READINESS probe, not a
    # mere liveness one: a server that bound its port but whose warmup failed (so
    # /predict would die) must report not-ready so a supervisor can recycle it.
    ready = {"v": warmup is None}
    if warmup is not None:
        print(f"[{backend}] warming up ...", flush=True)
        try:
            warmup()
            ready["v"] = True
        except Exception as exc:  # noqa: BLE001
            print(f"[{backend}] warmup failed (continuing): {exc}", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if ready["v"]:
                self._send(200, {"ok": True, "ready": True, "backend": backend})
            else:
                self._send(503, {"ok": False, "ready": False, "backend": backend})

        def do_POST(self):  # noqa: N802
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
            except Exception as exc:  # noqa: BLE001
                self._send(400, {"error": f"bad request: {exc}",
                                 "error_type": "invalid_input"})
                return
            task = req.get("task", "forward")
            try:
                if lock is not None:
                    with lock:
                        cands = predict(task, req)
                else:
                    cands = predict(task, req)
                self._send(200, {"backend": backend, "task": task,
                                 "candidates": cands})
            except NotImplementedError as exc:
                self._send(200, {"error": str(exc), "error_type": "unsupported"})
            except Exception as exc:  # noqa: BLE001
                self._send(200, {"error": f"{type(exc).__name__}: {exc}",
                                 "error_type": "upstream_error"})

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[{backend}] ready on http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()


__all__ = ["serve"]
