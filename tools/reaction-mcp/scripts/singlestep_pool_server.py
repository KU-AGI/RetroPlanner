#!/usr/bin/env python
"""Warm HTTP server exposing the POOL CONSENSUS single-step retro as an SSR expander.

`reaction_predict_singlestep_retro` fans out to every live retro backend and votes;
otherwise it runs in-process (MCP) or behind the pool_server contract. The
retrosynthesis drivers -- the board and the route search -- speak ONE single-step wire
contract, the one the root_aligned (SSR) replicas serve. This server speaks that same
contract, so the consensus becomes a drop-in swap for the expander with no change on the
driver side:

    POST /predict {"smiles": str, "top_n": int}
      -> [{"precursors": [smiles, ...], "confidence": float,
           "n_models": int, "models": [...]}, ...]        # ranked, best first
    GET  /health -> {"ok", "ready", "backend": "pool_consensus", "backends_live": [...]}

    RSMILES_SSR=http://127.0.0.1:8231/predict python scripts/traj_route_search.py ...

A transport/pool failure is NOT an empty candidate list. An empty list is the models'
answer and makes the molecule a dead end; a starved backend that returned [] would be
recorded as honest chemistry. So when fewer than ``--min-backends`` backends answered,
this replies 503 and both drivers turn that into PredictorUnavailable -> the target is
marked errored, not unsolved. ``--min-backends`` defaults to the number of retro
backends live at startup, so a 2-model (R-SMILES + LocalRetro) consensus stays a
2-model consensus for the whole run and a fleet that dies mid-run stops the run instead of quietly changing the arm.

Run in the reaction-mcp env, pointed at the replica proxies:

    source scripts/replica_urls.env      # written by launch_replicas.sh rsmiles|localretro
    REACTION_RETRO_DEPTH=20 python scripts/singlestep_pool_server.py --port 8231

REACTION_RETRO_DEPTH is the per-model fan-out (how many candidates each backend puts
into the vote); the request's ``top_n`` stays the number of consensus candidates
returned. The retro backends are the two in pool/config.py: rsmiles and localretro.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# import the reaction-mcp package (scripts/ is a sibling of reaction_mcp/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reaction_mcp import server as S  # noqa: E402
from reaction_mcp.pool.base import Task  # noqa: E402
from reaction_mcp.pool.registry import get_pool  # noqa: E402

STATE: dict = {"min_backends": 0, "live": [], "top_k": 10, "calls": 0, "aborts": 0,
               "degraded": 0}

# A backend can fail two ways, and only one of them is an outage. TransportError --
# connection refused, read timeout, non-JSON -- means the model never saw the molecule,
# and the pool labels it "upstream_error". Everything else (internal_error,
# invalid_input, unsupported) is the model ANSWERING that it cannot handle this input:
# a graph-edit model can, for instance, raise IndexError on the radical fragments another
# model emits as precursors ('C[O]', '[N]O'), reproducibly, while its server is healthy. Aborting a
# whole target because one intermediate is junk to one model would be wrong, so those
# count as answered-with-nothing and the call is recorded as degraded instead.
_OUTAGE = frozenset({"upstream_error"})
_LOCK = threading.Lock()


def live_retro_backends() -> list[str]:
    pool = get_pool()
    return [p.name for p in pool.predictors_for(Task.RETRO)]


def _rank_weight(cand: dict, idx: int) -> float:
    """One nonnegative weight per consensus candidate, later normalised to sum 1.

    The pool does not return a probability. A candidate carries ``consensus_score``
    (= cross-model approval votes + mean reciprocal rank, so >= 1 and unbounded above),
    not a calibrated likelihood, and the per-backend scores it is built from are on
    mutually incompatible scales (a template score such as localretro's 0.78 next to a
    beam log-probability such as -2.0 for the same disconnection) -- averaging them would be meaningless. Retro* costs an edge as
    -log(probability), so the arm needs SOME distribution over the returned candidates.

    Sum-normalising ``consensus_score`` is the mapping used: it reproduces the pool's own
    ranking exactly, is positive everywhere (so no zero-cost edge), and adds no
    calibration the pool did not claim. Consequence to keep in mind when reading the
    arm: consensus weights span a much narrower range than a seq2seq model's
    probabilities, so Retro* discriminates less sharply between siblings here than it
    does over root_aligned. The raw fields travel with every candidate, so a different
    mapping can be re-derived from a finished run without re-querying the models.
    """
    for key in ("consensus_score", "rrf_score", "votes"):
        v = cand.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return 1.0 / (idx + 1)


def predict(smiles: str, top_n: int) -> tuple[list[dict], dict]:
    """Consensus candidates in the SSR wire shape, plus the call's provenance."""
    out = S.reaction_predict_singlestep_retro(smiles, top_k=int(top_n), return_text=False)
    if not isinstance(out, dict) or out.get("error"):
        raise RuntimeError(str((out or {}).get("error"))[:200])
    ok = out.get("backends_ok") or []
    failed = out.get("backends_failed") or []
    unreachable = [b.get("backend") for b in failed if b.get("error_type") in _OUTAGE]
    refused = [b.get("backend") for b in failed if b.get("error_type") not in _OUTAGE]
    meta = {"backends_ok": ok, "unreachable": unreachable, "refused": refused,
            "abstained": bool(out.get("abstained")),
            "agreement": out.get("agreement"),
            "mass_rejected": out.get("mass_rejected")}
    if len(ok) + len(refused) < STATE["min_backends"]:
        raise RuntimeError(
            f"only {len(ok) + len(refused)}/{STATE['min_backends']} retro backends "
            f"responded (ok={ok}, refused={refused}, unreachable={unreachable})")
    if refused:
        with _LOCK:
            STATE["degraded"] += 1
    # An abstention (gate_tau not met) is a real answer -- no candidates -- but with
    # gate_tau=0 (the default) it never fires, so it is not silently a dead end here.
    cands = [c for c in (out.get("candidates") or []) if c.get("molecules")]
    weights = [_rank_weight(c, i) for i, c in enumerate(cands)]
    total = sum(weights) or 1.0
    preds = []
    for c, w in zip(cands, weights):
        preds.append({"precursors": [m for m in c["molecules"] if m],
                      # a normalised consensus weight, NOT a model probability -- see
                      # _rank_weight. Kept under the SSR contract's name so the drivers
                      # need no change; the raw consensus fields ride along beside it.
                      "confidence": w / total,
                      "n_models": c.get("n_backends"),
                      "models": c.get("backends"),
                      "consensus_score": c.get("consensus_score"),
                      "votes": c.get("votes"),
                      "agreement": c.get("agreement"),
                      "best_rank": c.get("best_rank")})
    return preds, meta


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # quiet: one line per 200 calls is enough, below
        pass

    def _send(self, code: int, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self._send(200, {"ok": True, "ready": True, "backend": "pool_consensus",
                         "min_backends": STATE["min_backends"],
                         "backends_live": STATE["live"],
                         "calls": STATE["calls"], "aborts": STATE["aborts"],
                         # calls a live backend refused this particular molecule, so the
                         # vote was over fewer than min_backends models
                         "degraded_calls": STATE["degraded"]})

    def do_POST(self):  # noqa: N802
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:  # noqa: BLE001
            self._send(400, {"error": f"bad request body: {exc}"})
            return
        smi = (req.get("smiles") or req.get("product_smiles")
               or req.get("product") or req.get("target") or "")
        if not smi:
            self._send(400, {"error": "no smiles in request"})
            return
        top_n = int(req.get("top_n") or req.get("top_k") or STATE["top_k"])
        try:
            preds, meta = predict(smi, top_n)
        except Exception as exc:  # noqa: BLE001
            with _LOCK:
                STATE["aborts"] += 1
            # 503, never [] -- see the module docstring.
            self._send(503, {"error": f"{type(exc).__name__}: {exc}"})
            print(f"[pool_consensus] ABORT {smi[:48]}: {type(exc).__name__}: "
                  f"{str(exc)[:160]}", flush=True)
            return
        with _LOCK:
            STATE["calls"] += 1
            n_calls = STATE["calls"]
        if n_calls % 200 == 0:
            print(f"[pool_consensus] {n_calls} calls, {STATE['aborts']} aborts, "
                  f"last ok={meta['backends_ok']}", flush=True)
        self._send(200, preds)


def main() -> None:
    ap = argparse.ArgumentParser(description="pool-consensus single-step retro server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8231)
    ap.add_argument("--top-k", type=int, default=10,
                    help="candidates returned when the request does not say")
    ap.add_argument("--min-backends", type=int, default=0,
                    help="abort a call that got fewer backends than this "
                         "(0 = however many are live at startup)")
    a = ap.parse_args()

    STATE["top_k"] = a.top_k
    STATE["live"] = live_retro_backends()
    STATE["min_backends"] = a.min_backends or len(STATE["live"])
    print(f"[pool_consensus] live retro backends: {STATE['live']}", flush=True)
    print(f"[pool_consensus] min_backends = {STATE['min_backends']} "
          f"(a call with fewer answering backends returns 503)", flush=True)
    if not STATE["live"]:
        raise SystemExit("no live retro backend; source scripts/replica_urls.env first")

    # warm the path once so the first search call does not pay import latency
    preds, meta = predict("CCO", a.top_k)
    print(f"[pool_consensus] warmup ok: {len(preds)} candidates, "
          f"backends={meta['backends_ok']}", flush=True)

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"[pool_consensus] serving on http://{a.host}:{a.port}/predict", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
