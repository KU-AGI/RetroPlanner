#!/usr/bin/env python
"""Shared vocabulary for the train-trajectory route search: paths, keys, stock.

Every script in the `traj_route_*` family imports this so that a route, a reaction and a
molecule mean the same thing at every stage — search, dedup, scoring, Pareto. Three of those
identities are the whole reason this module exists rather than being inlined:

  rxn_key(product, reactants)   reaction_mcp.scoring.rxn_key, the cache key of every axis.
                               The objective caches under data/node_scores/ are
                               shared with evaluation, so a reaction scored here is the SAME
                               cache entry evaluation reads. Change the key and every reaction
                               is re-scored and the two stop being comparable.

  set_key(steps)               frozenset of (product, sorted reactants) — `three_tables.rkey`.
                               Two routes with the same disconnections in any order are ONE.

  seq_key(steps)               the ordered tuple. Dedup for this dataset is order-SENSITIVE
                               (different order = different route), so this is the default.
                               A retrosynthesis route is a tree, so step ORDER is an artifact
                               of how the extractor walked the graph, not chemistry. Both keys
                               are therefore always recorded and both counts printed; n_seq /
                               n_set says how much of the order-sensitive diversity is real.

Stock. A leaf is purchasable or it is not, and the predicate has to be ONE object across
search (where it stops the recursion), scoring (where an unpriced leaf drops a route) and any
SFT rendering (where the observation says "buyable"). `Stock` loads one file in one of three
matching modes and is passed around explicitly; nothing here guesses a default.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

# config/paths.py, loaded by file under its own name rather than put on sys.path as
# `paths`, where any other module of that name would shadow it.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "rp_paths", str(Path(__file__).resolve().parents[3] / "config" / "paths.py"))
RP = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(RP)

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

SD = Path(__file__).resolve().parent
ROOT = SD.parent                                   # tools/reaction-mcp
REPO = ROOT.parents[1]                             # repo root
TRAJ = ROOT / "data" / "trajectories"
CLEAN = ROOT / "data" / "train_clean"
AIZ = ROOT / "models" / "aizynthfinder"

OUT = ROOT / "data" / "route_search"               # this pipeline's own outputs
RUNS = OUT / "runs"

# The axis scorers are reaction_mcp.scoring; tools/reaction-mcp goes on sys.path so every
# script here imports the same implementation the board does, and reads the same caches.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from reaction_mcp.scoring import NODE_SCORES  # noqa: E402
from reaction_mcp.scoring import rxn_key as _scoring_rxn_key  # noqa: E402

TRAIN_TRAJ = {
    "paroutes": TRAJ / "train_trajectories_multi_paroutes.jsonl",
    "fusionretro": TRAJ / "train_trajectories_multi_fusionretro.jsonl",
}
# The leakage set: ik14 of every test target (PaRoutes n1 and n5, FusionRetro test) that no
# train target or intermediate may be.
EXCL_IK14 = CLEAN / "test_excl_ik14.txt"

# Live single-step models on the SSR wire contract: POST {smiles, top_n} ->
# [{precursors|reactants, confidence}]. Each is a replica fleet, so the port is a proxy, not a
# model (scripts/launch_ssr_fleet.sh for R-SMILES, launch_ssr_fleet_localretro.sh for
# LocalRetro; or localretro_server.py behind scripts/ssr_shim.py). Each model has its own named
# slot so a run is never tagged with the wrong single-step model.
SSR = {
    "rsmiles": os.getenv("RSMILES_SSR", f"http://127.0.0.1:{RP.PORT_RSMILES}/predict"),
    "localretro": os.getenv("LOCALRETRO_SSR", f"http://127.0.0.1:{RP.PORT_LOCALRETRO}/predict"),
}


# ------------------------------------------------------------------ chemistry keys
_canon: dict[str, str | None] = {}
_ik: dict[str, str | None] = {}
_ikf: dict[str, str | None] = {}


def canon(smi: str) -> str | None:
    if smi not in _canon:
        m = Chem.MolFromSmiles(smi)
        _canon[smi] = Chem.MolToSmiles(m) if m is not None else None
    return _canon[smi]


def tkey(smi: str) -> str:
    """Canonical target key; falls back to the raw string so a target is never dropped."""
    return canon(smi) or smi


class DrawCache:
    """Disk-backed {smiles: menu} for ONE (model, draw). Thread-safe, append-only.

    Validated on LOAD, not only on write: rdkit's valence rules differ between rdkit
    versions (one accepts `CC(C)(C)[Si](C)(C)(C)Cl`, a later one rejects it), so a cache
    written under one interpreter can poison a run under the other.

    `base` is READ-ONLY prior coverage, `path` is this run's writable DELTA. The menus of one
    (model, draw, top_k) are scattered over several files -- the shared cache plus every
    sharded run that was given its own -- and a fresh search wants to see all of them without
    paying for them. Copying them into each shard's writable file costs a full copy per
    shard and makes every flush rewrite the whole union, which is O(n^2) in bytes over a run.
    Splitting it means each shard loads the union once, writes only what IT discovered, and
    merge_draw_cache.py folds the deltas back afterwards.

    A base file is only ever the SAME (model, draw, top_k). Draw 1 is an independent draw of
    a stochastic sampler and the run tag records which draw it searched, so folding d1 into a
    d0 run would make that label false -- the caller passes the paths, and it is the caller's
    job to keep them one oracle.
    """

    @staticmethod
    def _load(path) -> dict:
        out = {}
        raw = json.load(open(path))
        for k, v in raw.items():
            keep = []
            for rs, sc in v:
                cr = [canon(x) for x in rs]
                if cr and all(x is not None for x in cr):
                    keep.append((tuple(sorted(cr)), float(sc)))
            out[k] = keep
        return out

    def __init__(self, path, readonly=False, base=()):
        # readonly is for SHARDED replays: N processes re-running a recorded search off
        # this cache each hold the whole dict in memory, so any one of them flushing
        # rewrites the shared file from its own view and drops whatever the others
        # added since they loaded. A replay's misses are rare (they are the entries the
        # loader's valence check rejected) and go to the live model anyway, so giving up
        # the write-back costs a few calls and removes the clobber.
        # With `base` the clobber cannot happen in the first place: nothing writes a shared
        # file, so a fresh search keeps its menus AND its reuse.
        self.path, self.d, self.lock, self.dirty = path, {}, threading.Lock(), 0
        self.readonly = readonly
        self.base: dict = {}
        for bp in base:
            if bp and os.path.exists(bp) and os.path.abspath(bp) != os.path.abspath(path or ""):
                for k, v in self._load(bp).items():
                    self.base.setdefault(k, v)
        if path and os.path.exists(path):
            self.d = self._load(path)

    def __len__(self):
        return len(self.d) + len(self.base)

    def stats(self) -> dict:
        return {"base": len(self.base), "delta": len(self.d), "path": self.path}

    def get(self, smi):
        m = self.d.get(smi)
        return self.base.get(smi) if m is None else m

    def put(self, smi, menu):
        with self.lock:
            if smi in self.base:
                return                      # already covered; the delta stays a delta
            self.d[smi] = menu
            self.dirty += 1
            # The flush rewrites the WHOLE file, so a fixed interval is O(n^2) in total bytes
            # and a large cache would spend a growing share of the run on I/O. Scaling the
            # interval with the cache size keeps the overhead roughly constant. Writes stay
            # atomic (tmp + replace), so a crash costs at most the unflushed window and the job
            # is resumable.
            if self.dirty >= max(200, len(self.d) // 200):
                self._flush()

    def flush(self):
        with self.lock:
            self._flush()

    def _flush(self):
        if not self.path or not self.dirty or self.readonly:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({k: [[list(rs), sc] for rs, sc in v] for k, v in self.d.items()}, fh)
        os.replace(tmp, self.path)
        self.dirty = 0


EMPTY_RETRIES = [0]        # asks that came back 200-with-[] and were retried
EMPTY_BELIEVED = [0]       # molecules still empty after empty_retry tries


def ssr_call(url, smi, top_k, timeout, attempts=3, empty_retry=6):
    """POST {smiles, top_n} -> [(sorted canonical reactants, confidence)].

    Raises on transport failure rather than returning []: an empty list is the MODEL's answer
    ("no disconnection for this molecule") and makes the molecule a chemical dead end, so a
    dead backend returning [] is a silent, plausible-looking wrong result.

    AN EMPTY 200 IS NOT THE MODEL'S ANSWER EITHER. R-SMILES can answer HTTP 200 with `[]`
    for a molecule it answers with a full menu on the next call: its augmented draws can
    leave no valid precursor, so it is per-CALL and not deterministic, and nothing in the
    stack reports a failure. Taken at face value the molecule becomes a chemical dead end
    and the route is silently lost.

    So an empty result is RETRIED up to `empty_retry` times before it is believed. A
    molecule that really has no disconnection costs `empty_retry` fast calls, which is the
    price of not fabricating dead ends. `eval_board_agent.py:_fetch` carries the same guard
    for the board path; this is the search path.
    """
    body = json.dumps({"smiles": smi, "top_n": int(top_k)}).encode()
    last = None
    empty_tries = 0
    # An empty answer must not consume a transport attempt, or a molecule that answers []
    # three times in a row would raise instead of being retried.
    for i in range(attempts + empty_retry):
        try:
            req = urllib.request.Request(url, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as fh:
                out = json.loads(fh.read())
            if isinstance(out, dict) and out.get("error"):
                raise RuntimeError(str(out["error"])[:120])
            cands = out.get("candidates", out) if isinstance(out, dict) else out
            res = []
            for c in cands or []:
                rs = c if not isinstance(c, dict) else (c.get("precursors")
                                                       or c.get("reactants"))
                if isinstance(rs, str):
                    rs = rs.split(".")
                if not rs:
                    continue
                cr = [canon(x) for x in rs]
                if any(x is None for x in cr):
                    continue
                sc = float((c or {}).get("confidence") or 0.0) if isinstance(c, dict) else 0.0
                res.append((tuple(sorted(cr)), sc))
            # deduplicate keeping best score: the pooled backends can return one precursor set
            # twice under different orders, and two AndNodes for one disconnection double-count
            # the budget and the route set.
            best = {}
            for rs, sc in res:
                if rs not in best or sc > best[rs]:
                    best[rs] = sc
            menu = sorted(best.items(), key=lambda x: -x[1])
            if menu or empty_tries >= empty_retry:
                return menu
            # 200 with nothing in it: ask again rather than record a dead end.
            empty_tries += 1
            EMPTY_RETRIES[0] += 1
            time.sleep(0.5)
            continue
        except Exception as e:                                  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    if last is None:
        EMPTY_BELIEVED[0] += 1
        return []           # empty_retry consecutive empties: believe it
    raise RuntimeError(f"SSR {url} failed after {attempts}: {type(last).__name__}: {last}")


def ik14(smi: str) -> str | None:
    if smi not in _ik:
        m = Chem.MolFromSmiles(smi)
        _ik[smi] = Chem.MolToInchiKey(m)[:14] if m is not None else None
    return _ik[smi]


def ikfull(smi: str) -> str | None:
    if smi not in _ikf:
        m = Chem.MolFromSmiles(smi)
        _ikf[smi] = Chem.MolToInchiKey(m) if m is not None else None
    return _ikf[smi]


def rxn_key(product: str, reactants) -> str:
    """The identity of a disconnection, the key of every per-reaction cache.

    Reactants sorted and deduplicated: the same disconnection reached by two branches arrives
    with its precursors in either order, and scoring it twice would put two different numbers
    on one reaction (the forward model is not order-invariant on its input string).
    """
    return _scoring_rxn_key(product, reactants)


def longest_linear_sequence(steps, target: str):
    """Reactions on the route's deepest branch (its critical path), or None without the target.

    Branches off one product run in parallel, so the deepest one, not the step count, sets the
    elapsed synthesis time. A molecule reappearing on its own path stops that path.
    """
    by_product: dict = {}
    for p, rs in steps:
        by_product.setdefault(canon(p) or p, []).append([canon(r) or r for r in rs])

    def depth(mol, path):
        if mol in path or len(path) > 60:
            return len(path)
        return max([len(path)] + [depth(r, path | {mol})
                                  for rs in by_product.get(mol, []) for r in rs])

    d = depth(canon(target) or target, frozenset())
    return d or None


def set_key(steps) -> frozenset:
    return frozenset((p, tuple(sorted(rs))) for p, rs in steps)


def seq_key(steps) -> tuple:
    return tuple((p, tuple(sorted(rs))) for p, rs in steps)


def leaves(steps) -> list[str]:
    """Molecules that are consumed but never produced — the route's starting materials."""
    prod = {p for p, _ in steps}
    out, seen = [], set()
    for _, rs in steps:
        for r in rs:
            if r not in prod and r not in seen:
                seen.add(r)
                out.append(r)
    return sorted(out)


def internal(steps) -> list[str]:
    return sorted({p for p, _ in steps})


# ------------------------------------------------------------------ stock
class Stock:
    """One purchasability predicate, loaded from one file in one matching mode.

    Modes, and why the choice is not cosmetic. InChI normalises tautomers and charges;
    canonical SMILES does not. A leaf written as one tautomer is in an InChIKey stock and
    absent from the SMILES projection of the same catalogue (e.g. `O=c1[nH]nc(O)...` can
    match by InChIKey and not by SMILES). So the mode is part of the result and is recorded
    in every run sidecar.

      ik14    first InChIKey block, 14 chars. Skeleton only: stereo- and charge-blind, the
              LOOSEST of the three. This is what the train-leaf stocks and the board
              environment answer `done` with, so it is the default for this pipeline.
      ikfull  the whole 27-char InChIKey. Retro* + AiZynthFinder convention (eMolecules).
      smiles  exact canonical SMILES. A strict floor; quote it as such, never as coverage.
    """

    ALIASES = {
        "train_paroutes": ("ik14", TRAJ / "train_paroutes_leaves_ik14.txt"),
        "train_fusionretro": ("ik14", TRAJ / "train_fusionretro_leaves_ik14.txt"),
        "emols": ("ikfull", AIZ / "emol_inchikeys.txt"),
        "retroagent_bb": ("ikfull", AIZ / "retroagent_bb_inchikeys.txt"),
        "zinc_ik14": ("ik14", AIZ / "zinc_inchikey14.txt"),
    }

    def __init__(self, spec: str):
        if spec in self.ALIASES:
            self.mode, path = self.ALIASES[spec]
        elif ":" in spec:
            self.mode, p = spec.split(":", 1)
            path = Path(p)
        else:
            raise SystemExit(
                f"unknown stock {spec!r}; use one of {sorted(self.ALIASES)} "
                f"or mode:/abs/path.txt with mode in ik14|ikfull|smiles")
        if self.mode not in ("ik14", "ikfull", "smiles"):
            raise SystemExit(f"unknown stock mode {self.mode!r}")
        self.spec, self.path = spec, Path(path)
        if not self.path.exists():
            raise SystemExit(f"stock file missing: {self.path}")
        with open(self.path) as fh:
            self.keys = frozenset(ln.strip() for ln in fh if ln.strip())
        self._proj = {"ik14": ik14, "ikfull": ikfull, "smiles": canon}[self.mode]
        self._memo: dict[str, bool] = {}

    def __len__(self):
        return len(self.keys)

    def has(self, smi: str) -> bool:
        if smi not in self._memo:
            k = self._proj(smi)
            self._memo[smi] = bool(k) and k in self.keys
        return self._memo[smi]

    def describe(self) -> dict:
        return {"spec": self.spec, "mode": self.mode, "path": str(self.path),
                "n_keys": len(self.keys)}


def load_exclusion() -> frozenset:
    """The test-target ik14 that no train target or intermediate may be."""
    if not EXCL_IK14.exists():
        raise SystemExit(
            f"{EXCL_IK14} missing — it is the ik14 union of the PaRoutes n1 + n5 and "
            "FusionRetro test targets")
    with open(EXCL_IK14) as fh:
        return frozenset(ln.strip() for ln in fh if ln.strip())


# ------------------------------------------------------------------ io
def read_jsonl(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    n = 0
    with open(tmp, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def norm_steps(steps):
    """-> [(product, [reactants sorted])], the one shape every stage reads."""
    return [(p, sorted(rs)) for p, rs in steps]
