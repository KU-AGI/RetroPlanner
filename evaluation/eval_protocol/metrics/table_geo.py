"""Main table: Success Rate, Routes, Better-Than-Ref, Plaus, RT and Price at a FIXED budget B.

R-SMILES is the single-step model for every arm. Each column is read off what an arm had
after B single-step expansions, so the comparison is budget-matched rather than
run-to-completion:

  search arms    MCTS / Retro*-0 / Retro* / MORetro* were re-run at the budget, or stamp
                 each route with the call it was found at.
  Retro-R1       one continuous rollout over several rounds; the pooled file stamps every
                 route within its run, and the runs are laid end to end (from_pooled).
  RetroAgent     restarts failed rollouts until a route is found; same treatment, charged
                 what each run actually spent (SPEND, built by runners/build_spend.py).
  RetroPlanner   ends each rollout at its first complete route and restarts, spending the
                 rest of the budget on alternative routes; every route carries the number of
                 unique molecules seen when it was found (from_board).

The budget unit is unique molecules: RetroAgent and Retro-R1 count only unique molecules and
keep no revisit record, so it is the one unit all arms share.

An arm with no file at this budget prints '--', never 0: "we did not measure it" and "it
scored nothing" are different claims.

Route per target: of the Pareto non-dominated routes on (plaus, rt, price), the one with the
highest (q_p + q_rt + u_c)/3, u_c = 1 - log(1+min(q_c, P_max))/log(1+P_max). The price scale
is fixed by P_max alone, so the same route gets the same score whichever planner returns it.
Routes reaching the same leaf set count as one.

IMPUTED QUALITY (IMPUTE=1). A target with no stock-gated route by B scores 0 / 0 / PRICE_CAP
and the denominator is every benchmark target. The cap also CLIPS the solved side, because
some representative routes cost more than it and without clipping, failing those targets
would score better than solving them. BTR is imputed the same way: its denominator is every
target that HAS a reference route. IMPUTE=0 reports quality over solved targets only. The
two tables answer different questions, so never publish only one.

Run from $RP_MCP (runners/table_main.sh does); the data paths are relative to it.

  BENCH=uspto|chembl   BUDGET=50|300   IMPUTE=1|0   PRICE_CAP=1000
  inputs: R1_<bench> RA_<bench> RP_<bench>, SPEND_DIR
"""
import json, glob, os, sys, math, statistics as st, collections, importlib.util, random, re
# config/paths.py, loaded by file location under its own name rather than as `paths`, where
# any other module of that name would shadow it.
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
    """A RUNNING sweep looks exactly like a finished one from the outside: the file simply has
    fewer lines. Dividing its solved count by the full benchmark size reports a partial run as
    a bad arm, so the record count is checked against the target count and a short file is
    refused rather than scored."""
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


# What each (target, run) of a pooled baseline actually spent. USPTO and ChEMBL share one file.
try: SPEND = json.load(open(f"{SPEND_DIR}/ra_spend_uspto.json"))
except Exception: SPEND = {}

random.seed(0)
NPERM = 20


def _run_id(src):
    if not (src and isinstance(src[0], list)): return "?"
    f0 = src[0]
    if "retro_r1" in f0[0]: return re.sub(r"\.r\d+\.jsonl$", "", f0[-1])   # rounds of one run
    return next((x for x in f0 if x.startswith("eval_")), json.dumps(f0[1:]))


def from_pooled(pat):                    # Retro-R1 / RetroAgent
    """A pooled baseline file is SEVERAL INDEPENDENT RUNS, and its found_at_calls is a WITHIN-RUN
    stamp. Cutting it at B keeps each run's first B calls, so the arm silently gets one budget B
    per run. The runs are therefore laid END TO END: run i starts where run i-1 stopped, each
    charged what it actually spent (SPEND), with the last route's stamp as a fallback where
    SPEND has no entry. Order is arbitrary, so NPERM shuffles are drawn and the median-sized
    result is taken. Rounds of one Retro-R1 run share a run id, so they stay one continuous run.

    Single-run pools (no run id) fall through to a plain stamp cut, which is already correct.
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
                byrun[r["target"]][_run_id(r.get("sources"))].append((s, float(fa)))
    if not hit: return None
    out = {}
    for t, runs_ in byrun.items():
        # Every run that EXECUTED on this target. byrun only holds runs with a route that passed
        # the gate, so runs that spent budget and failed would otherwise be free and the
        # cumulative axis would collapse toward within-run.
        runs = sorted(set(runs_) | set(SPEND.get(t, {})))
        if len(runs) <= 1:
            out[t] = [x for x, c in runs_.get(runs[0], []) if c <= B] if runs else []
            continue
        cost = {r: float(SPEND.get(t, {}).get(r, max((c for _, c in runs_.get(r, [])), default=0.0)))
                for r in runs}
        draws = []
        for _ in range(NPERM):
            order = runs[:]; random.shuffle(order)
            off = 0.0; keep = []
            for r in order:
                for x, c in runs_.get(r, []):
                    if off + c <= B: keep.append(x)
                off += cost[r]
            draws.append(keep)
        draws.sort(key=len)
        out[t] = draws[len(draws)//2]
    return {t: v for t, v in out.items() if v}


def from_board(pat):                     # RetroPlanner
    """Every route carries found_at_unique: unique molecules seen when it was found, accumulated
    across the restarts. It is None in resumed runs (the counter restarts and undercounts);
    those routes are dropped rather than stamped in another unit."""
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
    """Routes reaching the same LEAF SET count as one: a plan's value is what you buy to start
    from, and counting write-ups of paths to the same stock would report the arm that dug into
    one spot as the most diverse one."""
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
    # Dedup once, here: not only the Routes column but also the pool axes3 sees must shrink --
    # if POOLCAP fills with duplicates, even the representative route shifts.
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
        # GEOMETRIC MEAN, not the median. A median over an imputed column has a threshold:
        # once more than half the targets are unsolved the statistic IS the cap. The geometric
        # mean moves continuously with coverage, and log space is already this pipeline's
        # scale for price, chosen because leaf prices are heavy-tailed.
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
        # 10 USPTO runs each.
        ("Retro-R1",   from_pooled, os.environ.get(
            "R1_uspto", "data/route_search/pooled_retro_r1_rsmiles_t0_fa.jsonl")),
        ("RetroAgent", from_pooled, os.environ.get(
            "RA_uspto", "data/route_search/pooled_retroagent_rsmiles_k10_fa.jsonl")),
        ("RetroPlanner", from_board, os.environ.get("RP_uspto", "results/forced_iter_retroplanner_emols_div_t06_b300_ep40.jsonl")),
    ])
elif BENCH == "chembl":
    render("ChEMBL-1000 / R-SMILES  (no gold -> BTR not computable)", 1000, {}, [
        ("MCTS",     lambda p: from_run(p, 1000), f"{R}/chembl1000__rsmiles__mcts__b{B}__d0.jsonl"),
        ("Retro*-0", lambda p: from_run(p, 1000), f"{R}/chembl1000__rsmiles__retrostar__b{B}__d0.jsonl"),
        ("Retro*",   lambda p: from_run(p, 1000), f"{R}/chembl1000__rsmiles__retrostar-value__b{B}__d0.jsonl"),
        # The b500 run, sharded; routes are cut at B by their found_at_calls stamps.
        ("MORetro*", from_parallel, f"{R}/chembl1000__rsmiles__moretro-bo-sobol__b500__d0.shard*of350.jsonl"),
        # 4 ChEMBL runs.
        ("Retro-R1",   from_pooled, os.environ.get(
            "R1_chembl", "data/route_search/pooled_retro_r1_rsmiles_chembl1000_run14.jsonl")),
        ("RetroAgent", from_pooled, os.environ.get(
            "RA_chembl", "data/route_search/pooled_retroagent_rsmiles_chembl1000_fa.jsonl")),
        ("RetroPlanner", from_board, os.environ.get(
            "RP_chembl", "results/forced_iter_retroplanner_chembl1000_emols_div_t06_b300_ep40.jsonl")),
    ])
else:
    sys.exit(f"BENCH must be uspto or chembl, not {BENCH!r}")
