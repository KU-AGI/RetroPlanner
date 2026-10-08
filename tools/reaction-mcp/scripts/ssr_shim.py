#!/usr/bin/env python
"""Speak the SSR wire contract in front of pool_server-contract backends.

`traj_route_search.py` drives ONE contract -- SSR:

    POST /predict {"smiles": str, "top_n": int}
      -> [{"precursors": [smiles, ...], "confidence": float}, ...]

Most of this repo's single-step servers speak the other one, pool_server's:

    POST /predict {"task": "retro", "product_smiles": str, "top_k": int}
      -> {"backend":..., "task":..., "candidates": [{"molecules": [...], "score": float,
                                                     "rank": int}, ...]}

which is not a cosmetic difference: `ssr_call` sends no `task` field at all, pool_server
defaults a missing one to `"forward"`, and every retro backend then answers
"serves task=retro, got 'forward'" -- an error the search records as a dead expander.
This translates, so a pool-contract model can be searched with no change to the driver
and no change to the backend.

It load-balances too (least-outstanding, like pool_proxy), because pool_server holds a
global lock around predict(): one replica answers exactly one request at a time, so the
fleet, not the server, is the throughput knob.

    python ssr_shim.py --port 8620 --backends http://127.0.0.1:8097   # localretro_server.py
    LOCALRETRO_SSR=http://127.0.0.1:8620/predict   # whichever SSR key the run uses

A backend error is propagated, never turned into an empty candidate list: [] is the
MODEL's answer ("no disconnection") and makes the molecule a chemical dead end, so a
failing backend that returned [] would be recorded as honest chemistry.
"""
from __future__ import annotations

import argparse
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Backend:
    __slots__ = ("url", "inflight", "ok", "fail", "lock")

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        if not self.url.endswith("/predict"):
            self.url += "/predict"
        self.inflight = 0
        self.ok = 0
        self.fail = 0
        self.lock = threading.Lock()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--backends", required=True,
                    help="comma-separated pool-contract /predict URLs")
    ap.add_argument("--name", default="ssr_shim")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--max-inflight", type=int, default=0,
                    help="0 = unlimited; otherwise admit at most this many at once")
    a = ap.parse_args()

    pool = [Backend(u) for u in a.backends.split(",") if u.strip()]
    if not pool:
        raise SystemExit("no backends")
    sel = threading.Lock()
    gate = threading.Semaphore(a.max_inflight) if a.max_inflight > 0 else None

    def pick() -> Backend:
        with sel:
            b = min(pool, key=lambda x: x.inflight)
            b.inflight += 1
            return b

    def call(b: Backend, smiles: str, top_n: int):
        body = json.dumps({"task": "retro", "product_smiles": smiles,
                           "top_k": int(top_n)}).encode()
        req = urllib.request.Request(b.url, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=a.timeout) as fh:
            return json.loads(fh.read())

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):        # noqa: A003 - quiet; the run logs its own rate
            pass

        def _send(self, code: int, obj) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"ok": True, "ready": True, "backend": a.name,
                             "n_backends": len(pool),
                             "backends": [{"url": b.url, "inflight": b.inflight,
                                           "ok": b.ok, "fail": b.fail} for b in pool]})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except Exception:                                   # noqa: BLE001
                self._send(200, {"error": "bad json", "error_type": "bad_request"})
                return
            smiles = req.get("smiles") or req.get("product_smiles")
            top_n = int(req.get("top_n") or req.get("top_k") or 10)
            if not smiles:
                self._send(200, {"error": "no smiles", "error_type": "bad_request"})
                return
            if gate:
                gate.acquire()
            try:
                last = None
                for _ in range(a.retries):
                    b = pick()
                    try:
                        out = call(b, smiles, top_n)
                        with b.lock:
                            b.ok += 1
                    except Exception as e:                      # noqa: BLE001
                        last = e
                        with b.lock:
                            b.fail += 1
                        continue
                    finally:
                        with sel:
                            b.inflight -= 1
                    if isinstance(out, dict) and out.get("error"):
                        last = RuntimeError(str(out["error"])[:200])
                        continue
                    cands = out.get("candidates", out) if isinstance(out, dict) else out
                    res = []
                    for c in cands or []:
                        mols = (c.get("molecules") or c.get("precursors")
                                or c.get("reactants")) if isinstance(c, dict) else c
                        if isinstance(mols, str):
                            mols = mols.split(".")
                        if not mols:
                            continue
                        sc = float(c.get("score", c.get("confidence", 0.0)) or 0.0) \
                            if isinstance(c, dict) else 0.0
                        res.append({"precursors": list(mols), "confidence": sc})
                    self._send(200, res[:top_n])
                    return
                # Out of retries. An error, NOT [] -- see the module docstring.
                self._send(200, {"error": f"all backends failed ({last})",
                                 "error_type": "upstream_error"})
            finally:
                if gate:
                    gate.release()

    ThreadingHTTPServer.request_queue_size = max(512, 4 * (a.max_inflight or 128))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True
    print(f"[{a.name}] SSR on http://127.0.0.1:{a.port}/predict -> "
          f"{len(pool)} pool-contract backend(s)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
