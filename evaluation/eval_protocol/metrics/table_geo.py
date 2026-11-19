"""Main table at a FIXED budget B: Success Rate, Routes, Better-Than-Ref, Plaus, RT, Price.

R-SMILES everywhere; every column is read off what an arm had after B single-step expansions.
The budget unit is UNIQUE MOLECULES -- RetroAgent (`seen_smiles`) and Retro-R1 (`mol_dict`)
count only unique molecules and keep no revisit record, and so does RetroPlanner's
`found_at_unique`, so it is the one unit all arms share.

Cut at B:  search arms -> re-run at the budget, or found_at_calls;  Retro-R1 / RetroAgent ->
independent runs laid end to end, charged by UNION (needs SPEND_DIR/expansions*.json from
runners/extract_expansions.py);  RetroPlanner -> found_at_unique.

Route per target: of the Pareto non-dominated routes on (plaus, rt, price), the one with the
highest (q_p + q_rt + u_c)/3, u_c = 1 - log(1+min(q_c, P_max))/log(1+P_max). Same leaf set =
one route. No file at this budget prints '--', never 0.

IMPUTE=1: no stock-gated route by B scores 0/0/PRICE_CAP over every benchmark target (BTR:
every target with a reference route); the cap also clips the solved side. IMPUTE=0 scores
solved targets only. Publish both.

Run from $RP_MCP (runners/table_main.sh does).

  BENCH=uspto|chembl   BUDGET=50|300   IMPUTE=1|0   PRICE_CAP=1000
  inputs: R1_<bench> RA_<bench> RP_<bench>, SPEND_DIR
"""
import json, glob, os, sys, math, statistics as st, collections, importlib.util, random, re
# config/paths.py, loaded by file location under its own name so no other `paths` shadows it.
_spec = importlib.util.spec_from_file_location(
    "rp_paths", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "config", "paths.py"))
_RP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_RP)
sys.path.insert(0, os.path.join(_RP.MULTISTEP, "scripts"))
import traj_route_common as C, feas_arm_diversity as D, feas_cost as FC
from feas_route_diversity import axes3, pareto_idx
import numpy as np

BENCH = os.environ.get("BENCH", "uspto")
B = int(os.environ.get("BUDGET", "300"))
CAP = float(os.environ.get("PRICE_CAP", "1000"))
IMPUTE = os.environ.get("IMPUTE", "1") != "0"
SPEND_DIR = os.environ.get("SPEND_DIR", "data/route_search/spend")
POOLCAP = 200
R = "data/route_search/runs"

stock = C.Stock("emols")
gate = lambda s: bool(D.leaves_of(s)) and all(stock.has(x) for x in D.leaves_of(s))
st_ = lambda v: [(p, sorted(x)) for p, x in (v or [])]
plaus, price = FC.PlausScorer(), FC.PriceScorer()
rt = FC.RtScorer(url=None)
for p in glob.glob("data/route_search/feas/rt_overlay_*.json"):
    try: rt.cache.update(json.load(open(p)))
    except Exception: pass


# ---------------------------------------------------------------------------- the four stamp shapes

def from_run(pat, need=None):            # b{B} run: the file IS the cut
    """A running sweep looks like a finished one with fewer lines, so a file short of `need`
    targets is refused rather than scored as a bad arm."""
    o = collections.defaultdict(list)
    seen = set()
    for f in sorted(glob.glob(pat)):
        for l in open(f):
            if not l.strip(): continue
            try: r = json.loads(l)
            except Exception: continue
            seen.add(r["target"])
            for x in (r.get("routes") or []):
                s = st_(x)
                if s and gate(s): o[r["target"]].append(s)
    if not o: return None
    if need is not None and len(seen) < need:
        print(f"    !! {pat.split('/')[-1]}: {len(seen)}/{need} targets -- incomplete, skipped", flush=True)
        return None
    return o


def from_parallel(pat):                  # MORetro*: routes[i] paired with found_at_calls[i]
    o = collections.defaultdict(list)
    for f in sorted(glob.glob(pat)):
        for l in open(f):
            if not l.strip(): continue
            try: r = json.loads(l)
            except Exception: continue
            fa = r.get("found_at_calls") or []
            for i, x in enumerate(r.get("routes") or []):
                if i >= len(fa) or fa[i] is None or fa[i] > B: continue
                s = st_(x)
                if s and gate(s): o[r["target"]].append(s)
    return o if o else None


# What each (target, run) of a pooled baseline spent. Run ids do not identify the run set
# (every RetroAgent experiment shares the eval_* numbering), so the spend file is keyed by
# the pool's filename rather than left to the caller. RA_SPEND_FILE overrides.
_POOL_SPEND = {
    "pooled_menu_match_fa.jsonl":    "ra_spend_menu_match.json",
    "pooled_menu_ctrl_fa.jsonl":     "ra_spend_menu_ctrl.json",
    "pooled_menusos_match_fa.jsonl": "ra_spend_menusos_match.json",
    "pooled_menusos_ctrl_fa.jsonl":  "ra_spend_menusos_ctrl.json",
}
_SPEND_F = os.environ.get("RA_SPEND_FILE") or _POOL_SPEND.get(
    os.path.basename(os.environ.get("RA_uspto",
                     "data/route_search/pooled_menu_match_fa.jsonl")), "ra_spend_uspto.json")
try: SPEND = json.load(open(f"{SPEND_DIR}/{_SPEND_F}"))
except Exception: SPEND = {}
print(f"  [spend] {_SPEND_F}  ({len(SPEND)} targets)", file=sys.stderr)

# Each pooled run's ORDERED unique expansions per target: what lets from_pooled charge a
# molecule once per target instead of once per run. Built by runners/extract_expansions.py
# (uspto) / extract_chembl.py (chembl).
#
# IT MUST COME FROM THE SAME RUN DIRECTORY AS THE POOL. `retroagent_rsmiles_t0` and
# `..._t0_k10` number their experiments alike, so a file from the wrong one passes every
# check below and charges the pool with another run set's molecules -- nearly every shared
# target gets a different sequence. Picked from the pool path for that reason;
# override both together with RA_uspto + POOL_EXPANSIONS.
_RA_POOL = os.environ.get("RA_uspto", "data/route_search/pooled_retroagent_rsmiles_k10_fa.jsonl")
_EXPF = os.environ.get("POOL_EXPANSIONS",
                       f"{SPEND_DIR}/expansions_chembl.json" if BENCH == "chembl"
                       else f"{SPEND_DIR}/expansions_k10.json" if "_k10" in _RA_POOL
                       else f"{SPEND_DIR}/expansions.json")
try: EXP = json.load(open(_EXPF))
except Exception as e:
    sys.exit(f"!! union accounting needs {_EXPF} ({e}). Build it with "
             f"runners/extract_expansions.py (uspto) / runners/extract_chembl.py (chembl).")
print(f"  [union] {_EXPF}  {EXP.pop('_meta', '(no _meta: provenance unknown)')}", file=sys.stderr)

RUN_ARM = {}
"""run id -> the arm that produced it, filled from each pooled row's own `sources` token."""


def _arm_of(run_id):
    """RUN_ARM's token wins; the id prefix is only a fallback. Run ids are NOT unique across
    arms -- RetroPlanner's ChEMBL experiments and RetroAgent's share the eval_* numbering."""
    return RUN_ARM.get(run_id) or (
        "retro_r1" if str(run_id).startswith("ncurve") else "retroagent")

random.seed(0)
NPERM = 20


def _run_id(src):
    if not (src and isinstance(src[0], list)): return "?"
    f0 = src[0]
    if "retro_r1" in f0[0]: return re.sub(r"\.r\d+\.jsonl$", "", f0[-1])   # rounds of one run
    return next((x for x in f0 if x.startswith("eval_")), json.dumps(f0[1:]))


def from_pooled(pat):                    # Retro-R1 / RetroAgent
    """The file is SEVERAL INDEPENDENT RUNS and found_at_calls is a WITHIN-RUN stamp, so a
    plain cut at B would give the arm one budget B per run. The runs are laid END TO END as one
    campaign per target; order is arbitrary, so NPERM shuffles are drawn and the median-sized
    result taken. Rounds of one Retro-R1 run share a run id and stay one run; single-run pools
    fall through to the plain stamp cut.

    UNION ACCOUNTING, the only branch. A route stamped `c` in run r was found when r had opened
    its first `c` unique molecules, so it costs (molecules unique to the runs already laid
    down) + (how many of those first `c` are new to the campaign). Summing per-run costs paid
    for a molecule twice -- not the rule any arm applies to itself.

    FILTERED BY RUN-ID FAMILY: SPEND holds both arms in one dict (eval_* = RetroAgent,
    ncurve_* = Retro-R1), and almost every target carries both, so unfiltered this arm would
    pay the other one's runs.
    """
    byrun = collections.defaultdict(lambda: collections.defaultdict(list))
    hit = False
    for f in sorted(glob.glob(pat)):
        for l in open(f):
            if not l.strip(): continue
            try: r = json.loads(l)
            except Exception: continue
            fa = r.get("found_at_calls")
            if fa is None: continue
            hit = True
            s = st_(r.get("steps"))
            if s and gate(s):
                _rid = _run_id(r.get("sources"))
                byrun[r["target"]][_rid].append((s, float(fa)))
                _src = (r.get("sources") or [[None]])[0]
                RUN_ARM[_rid] = (_src[0] if isinstance(_src, list) and _src else None)
    if not hit: return None

    def _fam(r):
        r = str(r)
        return "eval" if r.startswith("eval_") else ("ncurve" if r.startswith("ncurve") else "other")
    _fams = {_fam(r) for t in byrun for r in byrun[t]} - {"other"}

    out = {}
    for t, runs_ in byrun.items():
        # Every run that EXECUTED here: byrun only holds runs with a gated route, so runs
        # that spent budget and failed would otherwise be free.
        runs = sorted(set(runs_) | {k for k in SPEND.get(t, {}) if _fam(k) in _fams})
        if len(runs) <= 1:
            out[t] = [x for x, c in runs_.get(runs[0], []) if c <= B] if runs else []
            continue
        draws = []
        for _ in range(NPERM):
            order = runs[:]; random.shuffle(order)
            seen, off, keep = set(), 0, []
            for r in order:
                # A run absent from EXP costs nothing, i.e. free budget. Refuse.
                _a = _arm_of(r)
                if _a not in EXP or r not in EXP[_a]:
                    sys.exit(f"!! union accounting: run {r!r} of arm {_a!r} is not in "
                             f"{_EXPF}. Extract its expansions, or point POOL_EXPANSIONS at "
                             f"a file that covers the pools being rendered.")
                seq = EXP[_a][r].get(t, [])
                newpos, n, loc = [], 0, set()
                for sm in seq:
                    if sm not in seen and sm not in loc:
                        n += 1
                    loc.add(sm); newpos.append(n)
                for x, c in runs_.get(r, []):
                    i = min(int(c), len(newpos)) - 1
                    if off + (newpos[i] if i >= 0 else 0) <= B:
                        keep.append(x)
                off += n
                seen |= set(seq)
            draws.append(keep)
        draws.sort(key=len)
        out[t] = draws[len(draws) // 2]
    return {t: v for t, v in out.items() if v}


def from_board(pat):                     # RetroPlanner
    """found_at_unique = unique molecules seen when the route was found, accumulated across
    restarts. None in resumed runs (the counter restarts), so those routes are dropped rather
    than stamped in another unit."""
    o = collections.defaultdict(list)
    hit = False; miss = 0
    for f in sorted(glob.glob(pat)):
        for l in open(f):
            if not l.strip(): continue
            try: r = json.loads(l)
            except Exception: continue
            for a in ((r.get("routes_union") or []) + (r.get("routes") or [])):
                fa = a.get("found_at_unique")
                if fa is None:
                    if a.get("found_at_calls") is not None: miss += 1
                    continue
                hit = True
                if fa > B: continue
                s = st_(a.get("steps"))
                if s and gate(s): o[r["target"]].append(s)
    if miss:
        print(f"    (!) {pat.split('/')[-2] if '/' in pat else pat}: "
              f"excluded {miss} routes without found_at_unique", file=sys.stderr)
    return o if hit else None


# ---------------------------------------------------------------------------- scoring

def _dedup_pool(rs):
    """Same LEAF SET = one route: a plan's value is what you buy to start from, so counting
    paths to the same stock would make the arm that dug in one spot the most diverse."""
    seen = {}
    for s_ in rs:
        seen.setdefault(frozenset(D.leaves_of(s_)), s_)
    return list(seen.values())


def _ulog(negcost):
    """u_c = 1 - log(1+min(q_c, CAP)) / log(1+CAP).  Unpriced is 0 (worst)."""
    if negcost is None:
        return 0.0
    c = min(max(-float(negcost), 0.0), CAP)
    return 1.0 - math.log1p(c) / math.log1p(CAP)


def score(a):
    """s(route) = (u_p + u_rt + u_c) / 3, the composite every route is judged by."""
    return ((a[0] or 0.0) + (a[1] or 0.0) + _ulog(a[2])) / 3.0


def pick_idx(A):
    """The representative route: highest s among the Pareto front."""
    if len(A) < 2:
        return 0
    idx = pareto_idx(A) or list(range(len(A)))
    return max(idx, key=lambda i: score(A[i]))


def row(nm, pool, N, GOLD):
    if pool is None:
        return dict(name=nm, solve=None)
    # Dedup here so the pool axes3 sees shrinks too: if POOLCAP fills with duplicates, even
    # the representative route shifts.
    pool = {t: _dedup_pool(v) for t, v in pool.items()}
    P, Rr, Pr, NR = [], [], [], []
    btr = 0; n_gold_solved = 0
    for t, rs in pool.items():
        if not rs: continue
        A = axes3(rs[:POOLCAP], plaus, price, rt)
        a = A[pick_idx(A)]
        if a[0] is not None: P.append(a[0])
        if a[1] is not None: Rr.append(a[1])
        if a[2] is not None: Pr.append(-a[2])
        NR.append(len(rs))
        if GOLD.get(t):
            n_gold_solved += 1
            # Better-Than-Ref: the representative route strictly beats the best reference
            # route on the same composite s; a tie is a loss.
            if score(a) > max(score(g) for g in axes3(GOLD[t], plaus, price, rt)): btr += 1
    ns = sum(1 for v in pool.values() if v)
    DEN  = N if IMPUTE else max(len(P), 1)
    DENR = N if IMPUTE else max(len(Rr), 1)
    DENB = (len(GOLD) if IMPUTE else max(n_gold_solved, 1)) if GOLD else None
    return dict(name=nm, solve=100*ns/N, routes=st.mean(NR) if NR else 0.0,
        btr=(100*btr/DENB if DENB else None),
        P=100*sum(P)/DEN, Rr=100*sum(Rr)/DENR,
        # GEOMETRIC MEAN, not the median: a median over an imputed column IS the cap once
        # over half the targets are unsolved. Log space is already the price scale here.
        Pr=(math.exp((sum(math.log(max(min(x, CAP), 1e-9)) for x in Pr)
                      + math.log(CAP) * (N - len(Pr))) / N) if IMPUTE
            else (math.exp(sum(math.log(max(x, 1e-9)) for x in Pr) / len(Pr))
                  if Pr else float("nan"))),
        Prmed=st.median(Pr) if Pr else float("nan"))


def render(title, N, GOLD, arms):
    mode = f"imputed 0/0/{int(CAP)}" if IMPUTE else "SOLVED-ONLY (no imputation)"
    print(f"\n{'='*94}\n{title}   (budget B={B} [unique mols], {mode})\n{'='*94}")
    print(f"{'Method':18}{'Success Rate':>14}{'Routes':>9}{'Better-Than-Ref':>17}"
          f"{'Plaus>=0.05':>13}{'RT<=5':>9}{'Price geo':>11}{'(solved med)':>11}")
    print("-"*94)
    for nm, fn, pat in arms:
        r = row(nm, fn(pat) if (fn and pat) else None, N, GOLD)
        if r.get("solve") is None:
            print(f"{nm:18}{'--':>14}{'--':>9}{'--':>17}{'--':>13}{'--':>9}{'--':>11}{'--':>11}")
        else:
            b = "N/A" if r["btr"] is None else f"{r['btr']:.2f}%"
            print(f"{nm:18}{r['solve']:>13.2f}%{r['routes']:>9.2f}{b:>17}"
                  f"{r['P']:>12.2f}%{r['Rr']:>8.2f}%{r['Pr']:>11.2f}{r['Prmed']:>11.2f}")
        sys.stdout.flush()
    print("="*94)


# ---------------------------------------------------------------------------- the two tables

if BENCH == "uspto":
    GOLD = {}
    for r in (json.loads(l) for l in open("data/route_search/targets_uspto190.jsonl") if l.strip()):
        for s0 in (r.get("gold_routes") or []):
            s = st_(s0)
            if s and gate(s): GOLD.setdefault(r["target"], []).append(s)
    render("USPTO-190 / R-SMILES", 190, GOLD, [
        ("MCTS",       lambda p: from_run(p, 190), f"{R}/uspto190__rsmiles__mcts__b{B}__d0.jsonl"),
        ("Retro*-0",   lambda p: from_run(p, 190), f"{R}/uspto190__rsmiles__retrostar__b{B}__d0.jsonl"),
        ("Retro*",     lambda p: from_run(p, 190), f"{R}/uspto190__rsmiles__retrostar-value__b{B}__d0.jsonl"),
        ("MORetro*",   lambda p: from_run(p, 190), f"{R}/uspto190__rsmiles__moretro-bo-sobol__b{B}__d0*.jsonl"),
        ("Retro-R1",   from_pooled, os.environ.get(
            "R1_uspto", "data/route_search/pooled_retro_r1_rsmiles_t0_fa.jsonl")),
        # THE MATCHED-MENU RUN: handed RetroPlanner's own score-top-k instead of its own
        # candidate list (menus of equal size), so the arms differ in the planner, not the
        # board. ONE experiment -- plain stamp cut, Routes not comparable with a multi-run
        # pool's, charged out of ra_spend_menu_match.json.
        ("RetroAgent", from_pooled, os.environ.get(
            "RA_uspto", "data/route_search/pooled_menu_match_fa.jsonl")),
        ("RetroPlanner", from_board, os.environ.get("RP_uspto", "results/forced_iter_retroplanner_emols_div_t06_b300_ep40.jsonl")),
    ])
elif BENCH == "chembl":
    render("ChEMBL-1000 / R-SMILES  (no gold -> BTR not computable)", 1000, {}, [
        ("MCTS",     lambda p: from_run(p, 1000), f"{R}/chembl1000__rsmiles__mcts__b{B}__d0.jsonl"),
        ("Retro*-0", lambda p: from_run(p, 1000), f"{R}/chembl1000__rsmiles__retrostar__b{B}__d0.jsonl"),
        ("Retro*",   lambda p: from_run(p, 1000), f"{R}/chembl1000__rsmiles__retrostar-value__b{B}__d0.jsonl"),
        # The b500 run, sharded; routes are cut at B by their found_at_calls stamps.
        ("MORetro*", from_parallel, f"{R}/chembl1000__rsmiles__moretro-bo-sobol__b500__d0.shard*of350.jsonl"),
        ("Retro-R1",   from_pooled, os.environ.get(
            "R1_chembl", "data/route_search/pooled_retro_r1_rsmiles_chembl1000_run14.jsonl")),
        ("RetroAgent", from_pooled, os.environ.get(
            "RA_chembl", "data/route_search/pooled_retroagent_rsmiles_chembl1000_fa.jsonl")),
        ("RetroPlanner", from_board, os.environ.get(
            "RP_chembl", "results/forced_iter_retroplanner_chembl1000_emols_div_t06_b300_ep40.jsonl")),
    ])
else:
    sys.exit(f"BENCH must be uspto or chembl, not {BENCH!r}")
