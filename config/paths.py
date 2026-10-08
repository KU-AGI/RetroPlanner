"""Python-side paths for the RetroPlanner tree. Import this instead of writing
path literals::

    import sys, os
    sys.path.insert(0, os.path.join(<repo root>, "config"))
    import paths as RP

Every root is derived from this file's own location or read from the
environment, so scripts work from any working directory and a different machine
only needs to export the roots. The shell counterpart is ``config/env.sh``; the
two agree on names (``RP_WORKTREE``, ``CONDA_ROOT``, ``RP_CACHE``).

This file intentionally knows nothing about *which* run produced *which* table.
That belongs to the component that owns it: ``evaluation/eval_protocol`` reads
its rows from ``rows/``.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVALUATION = os.path.join(ROOT, "evaluation")
MULTISTEP = os.path.join(ROOT, "tools", "reaction-mcp")

EVAL_PROTOCOL = os.path.join(EVALUATION, "eval_protocol")
BOARD = os.path.join(EVALUATION, "board")

# The scored per-arm tables the paper's tables are computed from (rows/, derived/),
# and the target list they are computed over.
ROWS = os.path.join(EVAL_PROTOCOL, "rows")
DERIVED = os.path.join(EVAL_PROTOCOL, "derived")
TARGETS_190 = os.path.join(EVAL_PROTOCOL, "targets_uspto190.jsonl")

# The heavy inputs (checkpoints, vendored repos, stocks, run output) sit in place under
# ROOT at the paths below, ignored by git. RP_WORKTREE names another tree only when
# config/import_worktree.sh imports them from it.
WORKTREE = os.environ.get("RP_WORKTREE", ROOT)
DATA = os.path.join(MULTISTEP, "data")
MODELS = os.path.join(MULTISTEP, "models")
# Per-reaction / per-molecule axis caches (plausibility, round-trip, price, ...).
NODE_SCORES = os.path.join(DATA, "node_scores")
RESULTS = os.path.join(MULTISTEP, "results")
LOGS = os.path.join(MULTISTEP, "logs")
PROTOCOL_RESULTS = os.path.join(EVAL_PROTOCOL, "results")
# Served LLM, single-step (syntheseus) and forward (ReactionT5v2) weights.
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
MODEL_DIR = os.environ.get("RP_MODEL_DIR", os.path.join(CHECKPOINTS, "retroplanner"))
# Upstream checkouts: the R-SMILES repo the alternative single-step server imports.
BASELINE = os.path.join(ROOT, "external", "baseline")
# Shared data: gold routes, PaRoutes, USPTO splits, stocks.
SCI_DATA = os.path.join(ROOT, "external", "sci-data")

# Conda *envs* directory, not the conda prefix -- same convention as the
# launchers. Each retro backend pins its own torch and gets its own env.
CONDA_ROOT = os.environ.get("CONDA_ROOT", "/mnt/data/miniconda3/envs")
CACHE = os.environ.get("RP_CACHE", "/mnt/data/.cache")


def interpreter(env):
    """Absolute python for a named conda env, or ValueError.

    Checked rather than assumed: a sweep that launches against a moved env
    writes logs and no results, because every worker dies on a missing
    interpreter and the driver has nothing to collect.
    """
    p = os.path.join(CONDA_ROOT, env, "bin", "python")
    if not os.access(p, os.X_OK):
        raise ValueError(f"conda env {env!r} has no interpreter at {p}")
    return p


# Shared run settings -- the same names and defaults as config/env.sh, so a Python driver
# and a shell launcher agree on the machine layout. Override from the environment.
def _env_int(name, default):
    return int(os.environ.get(name, default))


GPUS = [g for g in os.environ.get("RP_GPUS", "0,1,2,3,4,5,6,7").split(",") if g]
PORT_LLM = _env_int("RP_PORT_LLM", 8300)
PORT_FORWARD = _env_int("RP_PORT_FORWARD", 8090)
PORT_MENU = _env_int("RP_PORT_MENU", 9019)
PORT_RSMILES = _env_int("RP_PORT_RSMILES", 9020)
PORT_LOCALRETRO = _env_int("RP_PORT_LOCALRETRO", 9021)
PORT_TEACHER = _env_int("RP_PORT_TEACHER", 8080)
WORKERS = _env_int("RP_WORKERS", 8)
NSHARD = _env_int("RP_NSHARD", 8)
