#!/usr/bin/env python
"""LocalRetro warm /predict server -- template-based single-step model for the planner.

Wraps LocalRetro (GNN atom/bond edit-scoring -> local-template decode -> precursors)
as a single-molecule HTTP server matching the reaction-mcp pool /predict contract.
Must run in the `localretro` conda env (torch2.4 / dgl2.4 / dgllife).

  CUDA_VISIBLE_DEVICES=0 $CONDA_ROOT/localretro/bin/python localretro_server.py --port 8097
  # or CPU:  CUDA_VISIBLE_DEVICES="" ... (small GNN, CPU is fine)

Model: LocalRetro_USPTO_50K.pth (trained on USPTO-50K).
"""
import os
import sys
import math

_SELF = os.path.dirname(os.path.abspath(__file__))
# config/paths.py, loaded by file under its own name rather than put on sys.path as
# `paths`, where any other module of that name would shadow it.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "rp_paths", os.path.join(_SELF, "..", "..", "..", "config", "paths.py"))
RP = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(RP)
# LocalRetro uses relative paths (../models, ../data) + its scripts on sys.path.
# Resolve from this checkout by default; retain an env override for deployments
# where the model repository lives elsewhere.
_LR = os.getenv(
    "REACTION_LOCALRETRO_REPO",
    os.path.join(RP.BASELINE, "LocalRetro", "scripts"),
)
os.chdir(_LR)
sys.path.insert(0, _LR)
sys.path.insert(0, os.path.dirname(_LR))     # LocalTemplate package
sys.path.insert(0, _SELF)                    # pool_server

import torch
import torch.nn as nn
import dgl
import pandas as pd
from dgllife.utils import smiles_to_bigraph
from rdkit import Chem
from rdkit.rdBase import DisableLog

DisableLog("rdApp.*")

from utils import init_featurizer, load_model, predict as _gnn_predict          # noqa: E402
from get_edit import combined_edit, get_bg_partition                            # noqa: E402
from LocalTemplate.template_decoder import read_prediction, decode_localtemplate  # noqa: E402
from pool_server import serve                                                    # noqa: E402

_DATASET = os.getenv("REACTION_LOCALRETRO_DATASET", "USPTO_50K")
_STATE: dict = {}


def _load() -> dict:
    if _STATE:
        return _STATE
    args = {
        "dataset": _DATASET,
        "config_path": "../data/configs/default_config.json",
        "data_dir": f"../data/{_DATASET}",
        "model_path": f"../models/LocalRetro_{_DATASET}.pth",
        "mode": "test",
        "batch_size": 16,
        "top_num": int(os.getenv("REACTION_LOCALRETRO_TOPNUM", "50")),
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    }
    init_featurizer(args)
    model = load_model(args)          # builds model + loads checkpoint (mode=test)
    model.eval()
    d = args["data_dir"]
    ATP = {r["Class"]: r["Template"] for _, r in pd.read_csv(f"{d}/atom_templates.csv").iterrows()}
    BTP = {r["Class"]: r["Template"] for _, r in pd.read_csv(f"{d}/bond_templates.csv").iterrows()}
    tif = pd.read_csv(f"{d}/template_infos.csv")
    TIF = {tif["Template"][i]: {"edit_site": eval(tif["edit_site"][i]),
                                "change_H": eval(tif["change_H"][i]),
                                "change_C": eval(tif["change_C"][i]),
                                "change_S": eval(tif["change_S"][i])} for i in tif.index}
    _STATE.update(args=args, model=model, ATP=ATP, BTP=BTP, TIF=TIF)
    print(f"[localretro] loaded {args['model_path']} on {args['device']} "
          f"({len(ATP)} atom / {len(BTP)} bond templates)", flush=True)
    return _STATE


def _canon(s: str) -> str:
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else s


def predict(task: str, request: dict) -> list[dict]:
    if task != "retro":
        raise NotImplementedError(f"localretro supports only 'retro', got '{task}'")
    st = _load()
    args, model = st["args"], st["model"]
    product = request.get("product_smiles") or request.get("product")
    top_k = int(request.get("top_k", 10))
    top_num = max(top_k * 4, 20)          # extract extra edits; many decode to None

    graph = smiles_to_bigraph(product, node_featurizer=args["node_featurizer"],
                              edge_featurizer=args["edge_featurizer"], add_self_loop=True)
    if graph is None:
        return []
    bg = dgl.batch([graph])
    with torch.no_grad():
        atom_logits, bond_logits, _ = _gnn_predict(args, model, bg)
    atom_logits = nn.Softmax(dim=1)(atom_logits)
    bond_logits = nn.Softmax(dim=1)(bond_logits)
    graphs, nodes_sep, edges_sep = get_bg_partition(bg)
    ptypes, psites, pscores = combined_edit(
        graphs[0], atom_logits[:nodes_sep[0]], bond_logits[:edges_sep[0]], top_num)

    seen: set = set()
    cands: list[dict] = []
    for i in range(len(ptypes)):
        pstr = "(%s, %s, %s, %.3f)" % (ptypes[i], psites[i][0], psites[i][1], pscores[i])
        try:
            mol, pred_site, template, template_info, score = read_prediction(
                product, pstr, st["ATP"], st["BTP"], st["TIF"])
            local_template = ">>".join(["(%s)" % s for s in template.split("_")[0].split(">>")])
            decoded = decode_localtemplate(mol, pred_site, local_template, template_info)
        except Exception:
            continue
        if not decoded:
            continue
        reactants = sorted({_canon(x) for x in str(decoded).split(".") if Chem.MolFromSmiles(x)})
        if not reactants:
            continue
        key = ".".join(reactants)
        if key in seen:
            continue
        seen.add(key)
        cands.append({"molecules": reactants, "score": float(pscores[i]), "rank": len(cands),
                      "raw": {"template": template}})
        if len(cands) >= top_k:
            break
    return cands


def _warmup() -> None:
    predict("retro", {"product_smiles": "CC(=O)Oc1ccccc1C(=O)O", "top_k": 1})


if __name__ == "__main__":
    serve(predict, backend="localretro", default_port=8097, warmup=_warmup)
