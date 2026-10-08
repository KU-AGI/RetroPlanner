"""SCScore (Coley et al. 2018): synthetic complexity in [1, 5].

The Reaxys 1024-bit model, evaluated in numpy: a chiral Morgan radius-2 bit fingerprint
through the dense layers stored in models/scscore/*.as_numpy.json.gz (ReLU between layers),
mapped to [1, 5] by 1 + 4 * sigmoid.
"""
import gzip
import json
from typing import Optional

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from . import MODELS

RDLogger.DisableLog("rdApp.*")

WEIGHTS_DIR = MODELS / "scscore"
_layers = None
_memo: dict = {}


def _weights() -> list:
    global _layers
    if _layers is None:
        files = sorted(WEIGHTS_DIR.glob("*.as_numpy.json.gz"))
        if not files:
            raise FileNotFoundError(f"no SCScore weights in {WEIGHTS_DIR}")
        with gzip.open(files[0], "rt") as fh:
            _layers = [np.asarray(a, dtype=np.float32) for a in json.load(fh)]
    return _layers


def scscore(smiles: str) -> Optional[float]:
    """SCScore of one molecule, or None if it does not parse."""
    if smiles not in _memo:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            _memo[smiles] = None
        else:
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useChirality=True)
            x = np.zeros(1024, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(bv, x)
            w = _weights()
            for i in range(0, len(w), 2):
                x = x @ w[i] + w[i + 1]
                if i < len(w) - 2:
                    x = np.maximum(x, 0)
            _memo[smiles] = 1 + 4 / (1 + np.exp(-float(x[0])))
    return _memo[smiles]
