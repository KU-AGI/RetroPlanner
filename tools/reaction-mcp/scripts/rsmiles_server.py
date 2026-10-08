#!/usr/bin/env python
"""Warm server for R-SMILES (Zhong et al. 2022, Chemical Science).

Root-aligned SMILES augmented seq2seq retrosynthesis (template-free). Uses
OpenNMT-py 2.2.0 for translation.

Setup:
    # Uses the rsmiles conda env (OpenNMT-py 2.2.0); see docs/ENVIRONMENTS.md

    # 1. Download checkpoint and vocab from Google Drive:
    #    https://drive.google.com/drive/folders/1c15h6TNU6MSNXzqB6dQVMWOs2Aae8hs6
    #    Get: PtoR-50K-aug20 folder (checkpoint + vocab)
    pip install gdown
    gdown --folder 1c15h6TNU6MSNXzqB6dQVMWOs2Aae8hs6 -O /path/to/rsmiles_data/

    # 2. Set env vars pointing at checkpoint and vocab.

Usage:
    export REACTION_RSMILES_CHECKPOINT=/path/to/checkpoint/model_step_200000.pt
    export REACTION_RSMILES_VOCAB=/path/to/vocab.src
    python rsmiles_server.py --port 8100
    export REACTION_RSMILES_URL=http://127.0.0.1:8100/predict
    export REACTION_ENABLE_RSMILES=1

Env vars:
    REACTION_RSMILES_CHECKPOINT  — path to .pt checkpoint
    REACTION_RSMILES_VOCAB       — path to .src vocab file
    REACTION_RSMILES_DEVICE      — GPU id (default: 0, -1 for CPU)
    REACTION_RSMILES_AUG         — augmentation count (default: 20)
    REACTION_RSMILES_SEED        — 'smiles' (default, deterministic per molecule),
                                   'random' (unseeded shuffling), or any string
                                   to use as a salt for a different fixed draw
    REACTION_RSMILES_TOPK        — beam size / top-k (default: 10)
    REACTION_RSMILES_REPO        — R-SMILES repo (for preprocessing scripts)
    ONMT_TRANSLATE_BIN           — override onmt_translate binary path
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from pool_server import serve

# config/paths.py, loaded by file under its own name rather than put on sys.path as
# `paths`, where any other module of that name would shadow it.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "rp_paths", str(Path(__file__).resolve().parents[3] / "config" / "paths.py"))
RP = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(RP)

_REPO = Path(os.getenv(
    "REACTION_RSMILES_REPO",
    str(Path(RP.BASELINE) / "rsmiles"),
))
# Default to the checkpoint downloaded into the standard models dir
# (tools/reaction-mcp/models/rsmiles/USPTO_50K_PtoR.pt, the USPTO-50K
# product-to-reactant model). Override with REACTION_RSMILES_CHECKPOINT.
_DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "models" / "rsmiles" / "USPTO_50K_PtoR.pt"
_CKPT = os.getenv(
    "REACTION_RSMILES_CHECKPOINT",
    str(_DEFAULT_CKPT) if _DEFAULT_CKPT.exists() else "",
)
_VOCAB = os.getenv("REACTION_RSMILES_VOCAB", "")
_DEVICE = os.getenv("REACTION_RSMILES_DEVICE", "0")
_AUG = int(os.getenv("REACTION_RSMILES_AUG", "20"))
_TOPK_DEFAULT = int(os.getenv("REACTION_RSMILES_TOPK", "10"))

# onmt_translate from the OpenNMT env. Invoke via ``python -m onmt.bin.translate``
# rather than the console script: the installed ``bin/onmt_translate`` carries a
# absolute shebang that breaks if the env is moved ("bad interpreter"). The module
# entry point is shebang-independent.
#
# The env is `rsmiles` (OpenNMT-py 2.2.0) -- see tools/reaction-mcp/docs/ENVIRONMENTS.md.
# CONDA_ROOT keeps the prefix overridable.
_CONDA_ROOT = Path(RP.CONDA_ROOT)
_RETROG2S_ENV = _CONDA_ROOT / os.getenv("REACTION_RSMILES_ENV", "rsmiles")
_RETROG2S_PY = os.getenv(
    "REACTION_RSMILES_PYTHON",
    str(_RETROG2S_ENV / "bin" / "python")
    if (_RETROG2S_ENV / "bin" / "python").exists()
    else sys.executable,
)
# Back-compat: ONMT_TRANSLATE_BIN, if set, overrides with a single binary path.
_ONMT_BIN = os.getenv("ONMT_TRANSLATE_BIN", "")
_ONMT_CMD = [_ONMT_BIN] if _ONMT_BIN else [_RETROG2S_PY, "-m", "onmt.bin.translate"]

_LOCK = threading.Lock()

# Warm in-process OpenNMT translator (built once, reused across calls). This is
# the fast path: _run_onmt_translate spawns a fresh `onmt_translate` subprocess
# per request and reloads the checkpoint every call. Building the translator once
# and translating an in-memory src list keeps the model resident. The subprocess
# path is kept as a fallback.
_STATE: dict = {}


def _load_translator():
    """Build (once) and cache a warm OpenNMT-py translator for _CKPT."""
    if _STATE.get("translator") is not None:
        return _STATE["translator"]
    if not _CKPT or not os.path.exists(_CKPT):
        raise RuntimeError(
            f"R-SMILES checkpoint not found at {_CKPT!r}. "
            "Download from Google Drive and set REACTION_RSMILES_CHECKPOINT."
        )
    # Mask the GPU before torch initializes so the model lands on the right card;
    # onmt then sees a single device re-indexed to ordinal 0.
    if _DEVICE not in ("-1", ""):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", _DEVICE)

    import onmt.opts as opts
    from onmt.translate.translator import build_translator
    from onmt.utils.parse import ArgumentParser

    beam = max(_TOPK_DEFAULT, 10)
    onmt_args = [
        "-model", _CKPT,
        "-src", "/dev/null",
        "-output", "/dev/null",
        "-beam_size", str(beam),
        "-n_best", str(_TOPK_DEFAULT),
        "-max_length", "300",
        "-batch_size", "64",
        "-gpu", "0" if _DEVICE != "-1" else "-1",
    ]
    if _VOCAB:
        onmt_args += ["-src_vocab", _VOCAB, "-tgt_vocab", _VOCAB]
    parser = ArgumentParser()
    opts.config_opts(parser)
    opts.translate_opts(parser)
    opt = parser.parse_args(onmt_args)
    ArgumentParser.validate_translate_opts(opt)

    print(f"[rsmiles] building warm translator from {_CKPT} (gpu={_DEVICE}) ...", flush=True)
    translator = build_translator(opt, report_score=False)
    _STATE["translator"] = translator
    print("[rsmiles] warm translator ready — in-process inference", flush=True)
    return translator


def _translate_warm(tokenized_lines: list[str], top_k: int) -> list[tuple[str, float]]:
    """Translate an in-memory tokenized src list with the warm translator."""
    translator = _load_translator()
    _, all_preds = translator.translate(src=tokenized_lines, batch_size=64)
    out: list[tuple[str, float]] = []
    # all_preds: one list (n_best) of space-tokenized predictions per src line.
    for plist in all_preds:
        for pred in plist:
            smi = pred.replace(" ", "").strip()
            if smi:
                out.append((smi, 0.0))
    return out


def _smi_tokenizer(smi: str) -> str:
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    return " ".join(re.findall(pattern, smi))


def _root_align_smiles(smi: str, n_aug: int) -> list[str]:
    """Generate root-aligned SMILES augmentations.

    R-SMILES aligns the product root atom so the product and reactant SMILES
    share the same root. For a live server we approximate with random SMILES
    augmentation (canonical + random atom-order permutations) since the full
    root-alignment preprocessing requires the reactant side. This is good
    enough for inference: the model handles augmented SMILES at test time.
    """
    from rdkit import Chem
    import hashlib
    import random
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [smi] * n_aug
    # Seeded from the molecule, NOT from global RNG state. Unseeded shuffling makes the server
    # a different function on every call: the same product can return a different candidate
    # list each time (the tail more than the head), so a multi-step search over it is not
    # reproducible and two runs of the same algorithm can differ. Seeding per SMILES keeps
    # augmentation diverse ACROSS molecules, which is what it is for, while making a repeat
    # query on one molecule return what it returned before.
    # REACTION_RSMILES_SEED=random gives unseeded shuffling.
    _seed_mode = os.environ.get("REACTION_RSMILES_SEED", "smiles")
    if _seed_mode == "random":
        rng = random
    else:
        base = _seed_mode if _seed_mode not in ("smiles", "") else ""
        digest = hashlib.sha256((base + smi).encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
    results = [Chem.MolToSmiles(mol)]  # canonical first
    seen = set(results)
    for _ in range(n_aug * 3):
        perm = list(range(mol.GetNumAtoms()))
        rng.shuffle(perm)
        try:
            aug = Chem.MolToSmiles(Chem.RenumberAtoms(mol, perm), canonical=False)
        except Exception:
            aug = smi
        if aug not in seen:
            seen.add(aug)
            results.append(aug)
        if len(results) >= n_aug:
            break
    while len(results) < n_aug:
        results.append(smi)
    return results[:n_aug]


def _run_onmt_translate(tokenized_lines: list[str], top_k: int) -> list[tuple[str, float]]:
    """Run onmt_translate and return (smiles, score) pairs."""
    if not _CKPT or not os.path.exists(_CKPT):
        raise RuntimeError(
            f"R-SMILES checkpoint not found at {_CKPT!r}. "
            "Download from Google Drive and set REACTION_RSMILES_CHECKPOINT."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        src_file = os.path.join(tmpdir, "src.txt")
        out_file = os.path.join(tmpdir, "out.txt")

        with open(src_file, "w") as fh:
            for tok in tokenized_lines:
                fh.write(tok + "\n")

        cmd = [
            *_ONMT_CMD,
            "-model", _CKPT,
            "-src", src_file,
            "-output", out_file,
            "-n_best", str(top_k),
            "-beam_size", str(max(top_k, 10)),
            "-max_length", "300",
            "-batch_size", "8",
            # We mask the GPU via CUDA_VISIBLE_DEVICES below, so the device is
            # always re-indexed to ordinal 0 in onmt's view. Passing the raw id
            # (e.g. 3) here would raise "invalid device ordinal".
            "-gpu", "0" if _DEVICE != "-1" else "",
        ]
        # Remove empty -gpu arg if CPU
        cmd = [c for c in cmd if c]
        if _VOCAB:
            cmd += ["-src_vocab", _VOCAB, "-tgt_vocab", _VOCAB]

        env = os.environ.copy()
        if _DEVICE not in ("-1", ""):
            env["CUDA_VISIBLE_DEVICES"] = _DEVICE

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
            )
        except Exception as exc:
            print(f"[rsmiles] onmt_translate error: {exc}", flush=True)
            return []

        if result.returncode != 0:
            print(f"[rsmiles] onmt_translate failed: {result.stderr[-400:]}", flush=True)

        if not os.path.exists(out_file):
            return []

        with open(out_file) as fh:
            lines = [l.strip() for l in fh]

        # onmt_translate outputs n_best * n_src lines: first n_best for src[0], etc.
        # With augmentation, we have n_aug inputs; collect all predictions.
        preds_with_scores: list[tuple[str, float]] = []
        for line in lines:
            # De-tokenize: remove spaces between chars
            smi = line.replace(" ", "").strip()
            if smi:
                preds_with_scores.append((smi, 0.0))

        return preds_with_scores


def _canonicalize(smi: str) -> str | None:
    from rdkit import Chem
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def predict(task: str, request: dict) -> list[dict]:
    if task != "retro":
        raise NotImplementedError(f"rsmiles_server serves task=retro, got {task!r}")

    product = request.get("product_smiles") or ""
    top_k = int(request.get("top_k", _TOPK_DEFAULT))

    augmented = _root_align_smiles(product, _AUG)
    tokenized = [_smi_tokenizer(s) for s in augmented]

    with _LOCK:
        try:
            raw_preds = _translate_warm(tokenized, top_k)
        except Exception as exc:  # noqa: BLE001
            print(f"[rsmiles] warm translate failed ({type(exc).__name__}: {exc}); "
                  "falling back to subprocess", flush=True)
            raw_preds = _run_onmt_translate(tokenized, top_k)

    # De-duplicate and canonicalize
    seen: set[str] = set()
    candidates: list[dict] = []
    rank = 0
    for smi, score in raw_preds:
        canon = _canonicalize(smi)
        if canon is None or canon in seen:
            continue
        seen.add(canon)
        mols = [m for m in canon.split(".") if m]
        if mols:
            candidates.append({
                "molecules": mols,
                "score": score - rank * 0.01,  # approximate rank-order
                "rank": rank,
                "raw": {"backend": "rsmiles", "aug": _AUG},
            })
            rank += 1
        if rank >= top_k:
            break

    return candidates


def _warmup() -> None:
    if not _CKPT or not os.path.exists(_CKPT):
        print(f"[rsmiles] checkpoint not found at {_CKPT!r}, skipping warmup", flush=True)
        return
    _load_translator()  # build the warm translator before serving
    predict("retro", {"product_smiles": "CC(=O)Oc1ccccc1C(=O)O", "top_k": 1})


if __name__ == "__main__":
    serve(predict, backend="rsmiles", default_port=8100, warmup=_warmup)
