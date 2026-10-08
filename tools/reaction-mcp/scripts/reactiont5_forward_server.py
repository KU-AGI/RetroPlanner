#!/usr/bin/env python
"""ReactionT5v2 forward model over HTTP: the server behind the round-trip axis.

    python scripts/reactiont5_forward_server.py --port 8090

    GET  /  -> {"ok": true, "backend": "reactiont5v2_forward", ...}
    POST {"batch": [{"reactants_smiles": [...], "reagents_smiles": [...]}, ...], "top_k": k}
      -> {"results": [{"predictions": [{"rank": 1, "smiles": "..."}, ...]}, ...]}

The model reads `REACTANT:<a.b>REAGENT:<c>`, as it was trained. Beam outputs are
canonicalised, invalid and repeated products dropped, and the rest re-ranked from 1, so one
product never takes two of the top-k slots. A failure is an HTTP 500, never an empty list:
an empty list would read as "the product was not recovered".

REACTIONT5_MODEL (a local copy, set by config/env.sh, or the hub id), REACTIONT5_DEVICE,
REACTIONT5_BEAMS.
"""
import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

MODEL = os.environ.get("REACTIONT5_MODEL", "sagawa/ReactionT5v2-forward")
DEVICE = os.environ.get("REACTIONT5_DEVICE", "cuda:0")
BEAMS = int(os.environ.get("REACTIONT5_BEAMS", "10"))
BACKEND = "reactiont5v2_forward"

_state: dict = {}
_gen_lock = threading.Lock()


def load():
    if not _state:
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        _state["torch"] = torch
        _state["tok"] = AutoTokenizer.from_pretrained(MODEL)
        _state["model"] = T5ForConditionalGeneration.from_pretrained(MODEL).eval().to(DEVICE)
    return _state


def _canon(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def predict(batch: list, top_k: int) -> list:
    s = load()
    k = min(max(1, int(top_k)), BEAMS)
    prompts = ["REACTANT:" + ".".join(x for x in item.get("reactants_smiles") or [] if x)
               + "REAGENT:" + ".".join(x for x in item.get("reagents_smiles") or [] if x)
               for item in batch]
    enc = s["tok"](prompts, return_tensors="pt", padding=True, truncation=True,
                   max_length=400).to(DEVICE)
    with _gen_lock, s["torch"].no_grad():
        seqs = s["model"].generate(**enc, num_beams=BEAMS, num_return_sequences=BEAMS,
                                   max_length=150, do_sample=False)
    text = [s["tok"].decode(q, skip_special_tokens=True).replace(" ", "") for q in seqs]
    results = []
    for i in range(len(batch)):
        preds, seen = [], set()
        for raw in text[i * BEAMS:(i + 1) * BEAMS]:
            c = _canon(raw)
            if c is None or c in seen:
                continue
            seen.add(c)
            preds.append({"rank": len(preds) + 1, "smiles": c})
            if len(preds) == k:
                break
        results.append({"predictions": preds})
    return results


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                                     # noqa: A003
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):                                              # noqa: N802
        self._send({"ok": True, "backend": BACKEND, "model": MODEL})

    def do_POST(self):                                             # noqa: N802
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return self._send({"error": "bad json"}, 400)
        try:
            items = req["batch"] if "batch" in req else [req]
            res = predict(items, req.get("top_k", 5))
            self._send({"results": res, "backend": BACKEND, "n": len(items)}
                       if "batch" in req else {**res[0], "backend": BACKEND})
        except Exception as e:                                     # noqa: BLE001
            self._send({"error": f"{type(e).__name__}: {str(e)[:200]}"}, 500)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    load()
    print(f"[{BACKEND}] ready on http://{a.host}:{a.port} ({MODEL}, {DEVICE})", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
