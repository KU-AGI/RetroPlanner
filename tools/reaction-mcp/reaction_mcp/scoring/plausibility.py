"""AiZynthFinder's filter policy: P(feasible) for a (product, reactants) disconnection.

The model takes two 2048-long count fingerprints -- the product's, and the product's minus
the sum of the reactants' (the reaction difference fingerprint) -- built from Morgan radius
2 counts folded modulo 2048, as the policy was trained.
"""
import os
import threading
from typing import Optional, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from . import MODELS

RDLogger.DisableLog("rdApp.*")

MODEL = os.environ.get("REACTION_FILTER_MODEL",
                       str(MODELS / "aizynthfinder" / "uspto_filter_model.onnx"))
FP_LEN = 2048

_session = None
_session_lock = threading.Lock()
_fps: dict = {}


def _sess():
    """One CPU session, single-threaded and without spin-waiting.

    The model is a small MLP. onnxruntime's defaults (one thread per core, spinning between
    calls) buy it nothing and keep cores busy while idle, which a sharded run multiplies.
    The thread count changes scheduling only, not the values. ORT_INTRA_OP_THREADS=0
    restores the onnxruntime default.
    """
    global _session
    with _session_lock:
        if _session is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            n = int(os.environ.get("ORT_INTRA_OP_THREADS", "1"))
            if n > 0:
                so.intra_op_num_threads = n
                so.inter_op_num_threads = 1
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            _session = ort.InferenceSession(MODEL, sess_options=so,
                                            providers=["CPUExecutionProvider"])
    return _session


def _fp(smiles: str) -> Optional[np.ndarray]:
    if smiles not in _fps:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            _fps[smiles] = None
        else:
            counts = AllChem.GetMorganFingerprint(mol, 2).GetNonzeroElements()
            v = np.zeros(FP_LEN, dtype=np.float32)
            for bit, c in counts.items():
                v[bit % FP_LEN] += c
            _fps[smiles] = v
    return _fps[smiles]


def score_reactions(rxns: Sequence[tuple], batch: int = 512) -> list:
    """[(product, [reactants])] -> [P(feasible)], None where a SMILES does not parse."""
    rows, prod, diff = [], [], []
    for i, (p, rs) in enumerate(rxns):
        pf, rf = _fp(p), [_fp(r) for r in rs]
        if pf is None or any(f is None for f in rf):
            continue
        rows.append(i)
        prod.append(pf)
        diff.append(pf - np.sum(rf, axis=0))
    out: list = [None] * len(rxns)
    for s in range(0, len(rows), batch):
        y = _sess().run(None, {"input_1": np.stack(prod[s:s + batch]),
                               "input_2": np.stack(diff[s:s + batch])})[0].reshape(-1)
        for i, v in zip(rows[s:s + batch], y):
            out[i] = float(v)
    return out
