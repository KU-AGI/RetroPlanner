#!/usr/bin/env python
"""Serve the board's own menu cache first, and only ask the live model for what it lacks.

WHY THIS EXISTS. The corpus is rendered from `draw_cache/<model>__d0__k10.json` -- the menus
`render_board_episode.py` replays -- while `eval_board_agent*.py` takes only `--menu-url` and
asks a live single-step server for every molecule. So the model is trained on one candidate
list and evaluated against another, and the two disagree wherever the live model has drifted
from the cache or ranks the same reactions differently. A rank the board never showed is a
rank the model never saw, and a `done` on leaves the cached menu would not have produced comes
back as `illegal: is left to make and is not purchasable`.

The cache covers the products of the training corpus but not necessarily an eval target set, so a miss has to go somewhere: it goes upstream, to the same
SSR fleet the eval would otherwise have used alone. Hits are exact and free; misses cost one
live call and are counted, so `/health` says how much of a run was actually cache-backed.

    python menu_cache_proxy.py --port 9019 \\
        --cache ../data/route_search/draw_cache/rsmiles__d0__k10.json \\
        --upstream http://127.0.0.1:9020/predict

If --cache does not exist yet, the proxy starts empty, writes every upstream answer into
that file, and the next run reads it as an ordinary cache. An existing cache is only read:
it is the corpus's menu list, and eval misses must not be mixed into it.

Then point the eval at 9019 instead of 9020. The response shape is the upstream's --
`[{"precursors": [...], "confidence": float}, ...]` -- so nothing downstream changes.
"""
import argparse
import json
import os
import signal
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CACHE: dict = {}
UPSTREAM = ""
STATS = {"hit": 0, "miss": 0, "upstream_fail": 0}
LOCK = threading.Lock()
WRITE_PATH = ""                  # set only when --cache did not exist: the file being built
FLUSH_LOCK = threading.Lock()    # one writer of WRITE_PATH.tmp at a time
FLUSH_EVERY = 100


def from_cache(smiles: str, top_k: int):
    """Cache rows are [[reactants, confidence], ...]; the wire shape is the upstream's."""
    rows = CACHE.get(smiles)
    if rows is None:
        return None
    return [{"precursors": list(r), "confidence": float(q)} for r, q in rows[:top_k]]


def from_upstream(payload: dict):
    req = urllib.request.Request(
        UPSTREAM, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read().decode())


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):                                    # noqa: A003
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                                             # noqa: N802
        with LOCK:
            s = dict(STATS)
        tot = s["hit"] + s["miss"]
        self._send({"ok": True, "ready": True, "backend": "menu_cache+upstream",
                    "cache_entries": len(CACHE), "upstream": UPSTREAM,
                    "hit_rate": (s["hit"] / tot) if tot else None, **s})

    def do_POST(self):                                            # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(n).decode() or "{}")
        except Exception:                                         # noqa: BLE001
            return self._send({"error": "bad json"}, 400)
        smiles = payload.get("smiles")
        top_k = int(payload.get("top_k") or 10)
        # One SMILES per call is what the eval sends; a list is passed straight through so a
        # batched caller is never silently served a partial answer out of the cache.
        if isinstance(smiles, str):
            hit = from_cache(smiles, top_k)
            if hit is not None:
                with LOCK:
                    STATS["hit"] += 1
                return self._send(hit)
        with LOCK:
            STATS["miss"] += 1
        try:
            rows = from_upstream(payload)
            if WRITE_PATH and isinstance(smiles, str) and isinstance(rows, list) and rows:
                remember(smiles, rows)
            return self._send(rows)
        except Exception as e:                                    # noqa: BLE001
            with LOCK:
                STATS["upstream_fail"] += 1
            return self._send({"error": f"upstream: {type(e).__name__}: {e}",
                               "error_type": "upstream_error"}, 502)


def remember(smiles: str, rows: list) -> None:
    """Store an upstream answer in the cache's own row shape and flush every FLUSH_EVERY."""
    with LOCK:
        CACHE[smiles] = [[list(r.get("precursors") or []), float(r.get("confidence") or 0.0)]
                         for r in rows]
        due = len(CACHE) % FLUSH_EVERY == 0
    if due:
        flush()


def flush() -> None:
    # A failed write is logged, never raised: it would reach do_POST's except and turn an
    # answered request into an upstream_error.
    with FLUSH_LOCK:
        with LOCK:
            snap = dict(CACHE)
        tmp = WRITE_PATH + ".tmp"                 # tmp + replace: a killed flush leaves the old file
        try:
            with open(tmp, "w") as fh:
                json.dump(snap, fh)
            os.replace(tmp, WRITE_PATH)
        except OSError as e:
            print(f"# !! cache flush failed: {e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--upstream", required=True)
    a = ap.parse_args()
    global UPSTREAM, WRITE_PATH
    UPSTREAM = a.upstream
    if os.path.exists(a.cache):
        print(f"# loading {a.cache}", flush=True)
        CACHE.update(json.load(open(a.cache)))
    else:
        print(f"# {a.cache} not found -- building it from upstream answers", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.cache)), exist_ok=True)
        WRITE_PATH = a.cache
    print(f"# {len(CACHE):,} cached products · upstream {UPSTREAM}", flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"# listening on 127.0.0.1:{a.port}", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))     # turn kill into the finally below
    try:
        srv.serve_forever()
    finally:
        if WRITE_PATH and CACHE:
            flush()


if __name__ == "__main__":
    main()
