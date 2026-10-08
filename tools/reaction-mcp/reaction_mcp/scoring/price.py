"""MolPrice: predicted ln(USD/mmol) of a molecule.

The model is MORetro's numpy port (external/MORetro/moretro/external/molprice.py) with the
weights and feature scaler MORetro ships (models/objectives/). `predict` returns 0.0 when
it cannot build a fingerprint, and 0.0 is a real price on this scale, so `predict_price`
checks the fingerprint first and returns None instead.
"""
import csv
import glob
import os
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from . import EXTERNAL, MCP_ROOT

MORETRO = EXTERNAL / "MORetro"
WEIGHTS = MORETRO / "models" / "objectives" / "model_price.pkl"
# MolPrice predictions already made (prices*.csv: smi_can, price) on the leaves it was
# run over, read before the model is asked.
PRICE_TABLES = MCP_ROOT / "data" / "molprice"

_model = None
_lock = threading.Lock()


def model():
    global _model
    with _lock:
        if _model is None:
            if str(MORETRO) not in sys.path:
                sys.path.insert(0, str(MORETRO))
            from moretro.external.molprice import MolPrice
            _model = MolPrice(Path(WEIGHTS))
    return _model


def predict_price(smiles: str) -> Optional[float]:
    """ln(USD/mmol), or None when the molecule has no fingerprint or the featuriser fails."""
    m = model()
    try:
        fp = m.smi_to_fp(smiles)
        if fp is None or not np.any(fp):
            return None
        return float(m.predict(smiles))
    except Exception:  # noqa: BLE001 -- e.g. SpacialScore divides by the heavy-atom count
        return None


def load_price_tables(directory=PRICE_TABLES) -> dict:
    """Every prices*.csv in `directory` -> {canonical SMILES: ln(USD/mmol)}; "Error" rows skipped."""
    out: dict = {}
    for f in sorted(glob.glob(os.path.join(str(directory), "prices*.csv"))):
        with open(f) as fh:
            for row in csv.DictReader(fh):
                try:
                    out[row["smi_can"]] = float(row["price"])
                except (KeyError, TypeError, ValueError):
                    pass
    return out
