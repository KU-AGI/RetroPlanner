#!/usr/bin/env python
"""Least-outstanding load-balancing proxy for reaction-pool warm servers.

Every backend built on ``pool_server.serve`` holds a GLOBAL LOCK around
``predict`` (most of these models are not thread-safe), so one server answers
exactly one request at a time. When N clients fan out concurrently the
requests queue behind that lock and per-call latency scales with N while the
machine sits mostly idle.

This proxy fronts several REPLICAS of the same backend under one URL: it speaks
the identical contract, so the pool's ``REACTION_<NAME>_URL`` can point here
with nothing else changed. Dispatch is **least-outstanding** (not round-robin):
these models have request-dependent latency (beam search over a big molecule
costs far more than a small one), so picking the replica with the fewest
in-flight requests keeps a slow call from head-of-line blocking a queue.

    python pool_proxy.py --port 9003 --backend localretro \
        --replicas http://127.0.0.1:9200/predict,http://127.0.0.1:9201/predict

    GET /health  -> {"ok", "backend", "replicas": [{url, inflight, ok, fail}], ...}
    POST <any>   -> forwarded verbatim to the chosen replica

A replica that errors or times out is retried on the next-best replica (up to
``--retries`` total attempts) and its failure counter is bumped; it is taken out
of rotation after ``--max-fails`` consecutive failures and probed back in by the
health loop. Standard library only, so it runs in any of the backend envs.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Replica:
    __slots__ = ("url", "inflight", "ok", "fail", "consec_fail", "disabled_until", "lock")

    def __init__(self, url: str) -> None:
        self.url = url
        self.inflight = 0
        self.ok = 0
        self.fail = 0
        self.consec_fail = 0
        self.disabled_until = 0.0
        self.lock = threading.Lock()

    def snapshot(self) -> dict:
        return {"url": self.url, "inflight": self.inflight, "ok": self.ok,
                "fail": self.fail, "disabled": time.time() < self.disabled_until}


class Balancer:
    def __init__(self, urls: list[str], max_fails: int, cooldown_s: float) -> None:
        self.replicas = [Replica(u) for u in urls]
        self.max_fails = max_fails
        self.cooldown_s = cooldown_s
        self.lock = threading.Lock()

    def pick(self, exclude: set[str]) -> Replica | None:
        """Least-outstanding among enabled replicas (ties -> fewest total calls)."""
        now = time.time()
        with self.lock:
            live = [r for r in self.replicas
                    if r.url not in exclude and now >= r.disabled_until]
            if not live:
                # everything cooling down: fall back to any not-yet-tried replica
                live = [r for r in self.replicas if r.url not in exclude]
                if not live:
                    return None
            best = min(live, key=lambda r: (r.inflight, r.ok + r.fail))
            best.inflight += 1
            return best

    def done(self, r: Replica, ok: bool) -> None:
        with self.lock:
            # clamp: inflight must never go negative, or this replica would win
            # every least-outstanding pick and absorb all traffic
            r.inflight = max(0, r.inflight - 1)
            if ok:
                r.ok += 1
                r.consec_fail = 0
            else:
                r.fail += 1
                r.consec_fail += 1
                if r.consec_fail >= self.max_fails:
                    r.disabled_until = time.time() + self.cooldown_s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--backend", default="proxy", help="label reported by /health")
    ap.add_argument("--replicas", required=True,
                    help="comma-separated replica /predict URLs")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="per-attempt upstream timeout (s)")
    ap.add_argument("--retries", type=int, default=2,
                    help="total attempts per request across distinct replicas")
    ap.add_argument("--max-fails", type=int, default=3,
                    help="consecutive failures before a replica is cooled down")
    ap.add_argument("--cooldown", type=float, default=60.0)
    ap.add_argument("--max-inflight", type=int, default=0,
                    help="admit at most N requests to the replicas at once "
                         "(0 = unlimited; the rest wait here). Past a backend's "
                         "GPU capacity extra concurrency SUBTRACTS throughput, "
                         "because its replicas time-slice the same "
                         "GPUs. Queueing at the proxy keeps every admitted "
                         "request running at full speed.")
    args = ap.parse_args()

    urls = [u.strip() for u in args.replicas.split(",") if u.strip()]
    if not urls:
        raise SystemExit("no replica URLs given")
    bal = Balancer(urls, args.max_fails, args.cooldown)
    gate = threading.Semaphore(args.max_inflight) if args.max_inflight > 0 else None
    started = time.time()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

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
            snaps = [r.snapshot() for r in bal.replicas]
            live = sum(1 for s in snaps if not s["disabled"])
            self._send(200 if live else 503, {
                "ok": bool(live), "ready": bool(live), "backend": args.backend,
                "n_replicas": len(snaps), "n_live": live,
                "uptime_s": round(time.time() - started, 1),
                "replicas": snaps,
            })

        def do_POST(self):  # noqa: N802
            try:
                n = int(self.headers.get("Content-Length", 0))
                payload = self.rfile.read(n) or b"{}"
            except Exception as exc:  # noqa: BLE001
                self._send(400, {"error": f"bad request: {exc}",
                                 "error_type": "invalid_input"})
                return
            if gate is not None:
                gate.acquire()
            try:
                self._dispatch(payload)
            finally:
                if gate is not None:
                    gate.release()

        def _dispatch(self, payload: bytes) -> None:
            tried: set[str] = set()
            last_err = "no replica available"
            for _ in range(max(1, args.retries)):
                r = bal.pick(tried)
                if r is None:
                    break
                tried.add(r.url)
                req = urllib.request.Request(
                    r.url, data=payload,
                    headers={"Content-Type": "application/json"}, method="POST")
                # Only the UPSTREAM call may be retried, and done() must run
                # exactly once per pick(). Writing the reply to the client is
                # kept out of this try: a broken pipe (the client gave up) raises
                # OSError, and counting that as an upstream failure would
                # decrement inflight a second time. Negative inflight would then
                # make that replica permanently win least-outstanding, so one
                # replica would serve everything while the others sat idle.
                body = None
                try:
                    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                        body = resp.read()
                except (urllib.error.URLError, OSError, TimeoutError) as exc:
                    last_err = f"{type(exc).__name__}: {exc}"
                finally:
                    bal.done(r, body is not None)
                if body is not None:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            # Upstream failure is reported the way the backends report errors
            # (HTTP 200 + error envelope) so the pool records a failed backend
            # rather than raising a transport exception mid-search.
            self._send(200, {"error": f"proxy: all replicas failed ({last_err})",
                             "error_type": "upstream_error"})

    # socketserver's default listen backlog is 5. A search fanning N workers out at once
    # opens N connections in the same instant, the accept queue overflows and the kernel
    # answers the surplus with RST -- which reaches the caller as "Connection reset by
    # peer" and looks like a dead model rather than a full queue. Size it above any
    # plausible --max-inflight.
    ThreadingHTTPServer.request_queue_size = max(512, 4 * args.max_inflight)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"[proxy:{args.backend}] http://{args.host}:{args.port} -> "
          f"{len(urls)} replica(s)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
