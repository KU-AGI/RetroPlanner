"""Round-trip: does a forward model, given the precursors, predict the product back?

Client of the forward server (scripts/reactiont5_forward_server.py):

    POST {"batch": [{"reactants_smiles": [...], "reagents_smiles": []}, ...], "top_k": k}
      -> {"results": [{"predictions": [{"rank": 1, "smiles": "..."}, ...]}, ...]}

A prediction matches when its largest fragment equals the product's, since a forward model
may return by-products joined with '.'.
"""
import itertools
import os
import threading
import time
from typing import Iterator, Optional, Sequence

import requests
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_rr = itertools.count()
_rr_lock = threading.Lock()
_major: dict = {}


def server_url() -> str:
    """REACTION_FORWARD_URL, read per call; a comma-separated list is rotated over."""
    spec = os.environ.get("REACTION_FORWARD_URL") or \
        f"http://127.0.0.1:{os.environ.get('RP_PORT_FORWARD', '8090')}"
    urls = [u.strip() for u in spec.split(",") if u.strip()]
    with _rr_lock:
        return urls[next(_rr) % len(urls)]


def major_product(smiles: Optional[str]) -> Optional[str]:
    """Canonical SMILES of the fragment with the most heavy atoms, or None."""
    if smiles is None:
        return None
    if smiles not in _major:
        best, size = None, -1
        for frag in filter(None, smiles.split(".")):
            mol = Chem.MolFromSmiles(frag)
            if mol is not None and mol.GetNumHeavyAtoms() > size:
                best, size = Chem.MolToSmiles(mol), mol.GetNumHeavyAtoms()
        _major[smiles] = best
    return _major[smiles]


def _results(rxns, top_k, batch, timeout, url, retries=4) -> Iterator[tuple[int, dict]]:
    """(index, result) per reaction. An unanswered chunk raises after `retries` attempts:
    a missing answer must never be recorded as "not recovered"."""
    for s in range(0, len(rxns), batch):
        chunk = rxns[s:s + batch]
        payload = {"batch": [{"reactants_smiles": list(rs), "reagents_smiles": []}
                             for _, rs in chunk], "top_k": top_k}
        endpoint = url or server_url()
        reply, err = None, None
        for attempt in range(retries):
            try:
                reply = requests.post(endpoint, json=payload, timeout=timeout).json()
                # An HTTP 200 carrying {"error": ...} (a proxy whose replicas failed) is not
                # an answer either.
                if not isinstance(reply, dict) or "results" not in reply or reply.get("error"):
                    raise RuntimeError(f"no results in reply: {str(reply)[:160]}")
                break
            except Exception as e:  # noqa: BLE001
                reply, err = None, e
                time.sleep(2 ** attempt)
        if reply is None:
            raise RuntimeError(f"{endpoint} did not answer {len(chunk)} reactions after "
                               f"{retries} attempts: {str(err)[:160]}")
        for j, res in enumerate(reply["results"]):
            if isinstance(res, dict):
                yield s + j, res


def roundtrip(rxns: Sequence[tuple], top_k: int = 5, batch: int = 64, timeout: float = 600,
              url_override: Optional[str] = None) -> list:
    """[(product, [reactants])] -> rank (1..top_k) at which the product comes back, or None."""
    out: list = [None] * len(rxns)
    for i, res in _results(rxns, top_k, batch, timeout, url_override):
        want = major_product(rxns[i][0])
        if want is None:
            continue
        for pred in res.get("predictions") or []:
            if major_product(pred.get("smiles", "")) == want:
                out[i] = int(pred.get("rank", 99))
                break
    return out
