"""Both ablation tables at one post-hoc call budget.

A route is kept if it existed by `budget` UNIQUE MOLECULES OPENED -- not total single-step
calls. A molecule already expanded is free the next time it comes up, which is the axis
RetroAgent's solved_at_budget and Retro-R1's graph size use, and the only one they can be
compared on (neither records revisits). The gap is not cosmetic: the LLM arms restart, so
they re-open 1.4-1.7x, while the one-pass policies re-open 1.02x -- cutting on total opens
would shorten our budget by 30-40% and leave the baselines untouched. The harness also
stamps `found_at_calls` if the other axis is ever wanted.

The LLM arms carry the unique stamp on every route (`found_at_unique`, incremented from a
`set` of opened SMILES in eval_board_agent_iterative.py). The scripted policies do not --
they run one pass instead of the restart loop -- so the stamp is rebuilt from the dump:
walk `applied_steps` in order, count distinct `mid`s seen so far (measured 1:1 with
distinct SMILES on these dumps, so it is the same quantity `found_at_unique` counts),
and map each route's steps onto it through the `<mol xx>SMILES` tags the board renders
carry. Where a molecule was opened more than once the EARLIEST application is taken, so the
policy arms are credited with their routes as early as possible -- the bound that favours
the baseline, which is the direction a baseline should be given.

Why the budget matters enough to be a flag: the arms do not rank the same way at every
budget. Single-route leads under a tight one and multi-route (parallel) under a loose one;
the ladder beats the no-analysis rung below ~80 calls and loses above it. Any single-budget
table is a slice, and which slice is a choice that has to be stated. The default slice is
B=50 -- one episode's allowance, the tight end where the arms are separated by what they
open first rather than by how long they are allowed to keep going.

Columns:

  Success     share of targets the arm returned at least one route for by the budget.
  Routes      distinct routes per target, counted by LEAF SET.
  1stCutDiff  share of route PAIRS that take a DIFFERENT first cut -- a different
              disconnection at the target itself, via `analyze_route_diversity.first_cut`.
              Structural diversity where it costs something, as opposed to two routes that
              agree on the opening move and differ only in how a shared fragment is worked
              up. Averaged over TARGETS, so a target contributing one pair weighs the same
              as one contributing a hundred. Printed as -- where the arm supplies fewer
              than `--min-pairs` pairs in total: the scripted policies reach two routes on
              13-22 of the 60 targets and never a third, so their cell is a coin-flip
              proportion over a handful of single pairs, conditioned on the rare targets
              where they found anything to compare -- not a diversity average, and not
              comparable with an arm scored over 500 pairs. The threshold is not
              load-bearing here: pair counts in this grid are 13, 22, then 205 and up, so
              anything in (22, 205] suppresses the same two cells.
  BTR         share of targets whose best route beats the published route on
              plaus_passfrac x rt5 x (1 - min(1, cost/1000)).

    python scripts/ablation_tables_at_budget.py --budget 50 --common --latex
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from itertools import combinations
from pathlib import Path

from rdkit import RDLogger
import importlib.util

RDLogger.DisableLog("rdApp.*")
_s = importlib.util.spec_from_file_location(
    "ard", str(Path(__file__).resolve().parent / "analyze_route_diversity.py"))
ard = importlib.util.module_from_spec(_s); _s.loader.exec_module(ard)

R = Path("eval_protocol/results")
L = Path("data/route_search")
COST_REF = 1000.0
GOLD = "labeled_gold_retroagent_rsmiles.jsonl"

EXPANSION = [("Random", "policy/random_k4_190.jsonl", "rq_policy_random_k4_cf190", "routes", True),
             ("Score-only", "policy/score_k4.jsonl", "rq_policy_score_k4_cf190", "routes", True),
             ("Single-Route", "v20_single_route_cf190/iter.jsonl",
              "rq_v20_single_route_cf190", "routes_union", False),
             ("Multi-Route (Sequential)", "v20_multi_seq_cf190/iter.jsonl",
              "rq_v20_multi_seq_cf190", "routes_union", False),
             ("Multi-Route (Parallel)", "v20_multi_par_cf190/iter.jsonl",
              "rq_v20_multi_par_cf190", "routes_union", False)]
# L2_bond and L2_rxn are ALTERNATIVE RENDERINGS of the DISCONNECTIONS block at the L2 level,
# added for review -- siblings of L2 (=L2_disc on disk), not further steps of the chain, so
# both take L1 as their parent and carry L2's flags.
LADDER = [(r"$L_0$", "L0", None), (r"$L_1$", "L1", "L0"), (r"$L_2$", "L2_disc", "L1"),
          (r"$L_2$ (bond)", "L2_bond", "L1"), (r"$L_2$ (rxn)", "L2_rxn", "L1"),
          (r"$L_{3\text{-div}}$", "L3_div", "L2_disc"),
          (r"$L_{3\text{-ord}}$", "L3_ord", "L2_disc"), (r"$L_3$", "L4", "L2_disc")]
FLAGS = {"L0": (0, 0, 0, 0, 0), "L1": (1, 0, 0, 0, 0), "L2_disc": (1, 1, 0, 0, 0),
         "L2_bond": (1, 1, 0, 0, 0), "L2_rxn": (1, 1, 0, 0, 0),
         "L2": (1, 1, 1, 0, 0), "L3_div": (1, 1, 1, 1, 0), "L3_ord": (1, 1, 1, 0, 1),
         "L4": (1, 1, 1, 1, 1)}
PARENT_LABEL = {"L0": "--", "L1": r"$L_0$", "L2_disc": r"$L_1$",
                "L2_bond": r"$L_1$", "L2_rxn": r"$L_1$",
                "L3_div": r"$L_2$", "L3_ord": r"$L_2$", "L4": r"$L_2$"}

UNRESOLVED = [0, 0]   # routes whose first cut ard could not locate, routes seen


def first_cut(rt, target):
    """`ard.first_cut`, with the misses counted instead of silently dropped.

    ard matches the target by exact string and has no fallback, so a dump that writes the
    root product in a different but equivalent SMILES returns None and that route leaves
    the pair statistics without a trace. Measured on this grid the miss rate is 0/1745
    (B=50) and 0/4061 (B=300), and a canonicalising variant with a root-by-construction
    fallback agrees with ard on every pair -- but the counter stays so a future dump that
    breaks the assumption says so instead of quietly shrinking the denominator.
    """
    UNRESOLVED[1] += 1
    v = ard.first_cut(rt, target)
    if v is None:
        UNRESOLVED[0] += 1
    return v


def leaves(rt):
    prod = {z[0] for z in rt["steps"]}
    return frozenset(x for z in rt["steps"] for x in z[1] if x not in prod)


def route_key(steps):
    return frozenset((z[0], tuple(sorted(z[1]))) for z in steps)


def composite(r):
    f1, f2, c = r.get("plaus_passfrac"), r.get("rt5"), r.get("cost_usd")
    if f1 is None or f2 is None:
        return None
    return f1 * f2 * (1.0 - min(1.0, (c if c is not None else COST_REF) / COST_REF))


def policy_stamps(rec):
    """Rebuild found_at_unique for a single-pass policy dump."""
    tags = {}
    for msg in (rec.get("harmony_messages") or []):
        for mm in re.finditer(r"<mol ([a-z0-9]+)>([^<\s]+)", msg.get("content") or ""):
            tags.setdefault(mm.group(1), mm.group(2))
    seen, call_at = set(), []
    for s in rec["applied_steps"]:
        seen.add(s["mid"])
        call_at.append(len(seen))
    first = {}
    for i, s in enumerate(rec["applied_steps"]):
        smi = tags.get(s["mid"])
        if smi is not None and smi not in first:
            first[smi] = call_at[i]
    out = {}
    for rt in (rec.get("routes") or []):
        idx = [first.get(z[0]) for z in rt["steps"]]
        out[route_key(rt["steps"])] = max(idx) if all(x is not None for x in idx) else None
    return out


def reference():
    best = {}
    for line in open(L / GOLD):
        r = json.loads(line)
        c = composite(r)
        if c is not None:
            best[r["target"]] = max(best.get(r["target"], 0.0), c)
    return best


_JSONL: dict[str, list] = {}


def _read(path):
    """Memoised jsonl load. A budget SWEEP calls row() once per budget on the same two
    files, and the labeled dumps run to 275MB, so re-reading them per budget is most of
    the wall-clock of a curve. Keyed by path; nothing here mutates the records."""
    k = str(path)
    if k not in _JSONL:
        _JSONL[k] = [json.loads(l) for l in open(path) if l.strip()]
    return _JSONL[k]


# A TARGET SUBSET, FOR MATCHED-SUBSET COMPARISONS. Left None, every row is over the arm's
# own full target list, which is what the paper tables use. Set to a set of SMILES, every
# arm is restricted to the same targets -- the only honest way to table an arm that was
# measured on a subset next to one measured on all 190, because the missing targets are
# never a random sample of the benchmark.
ONLY: set | None = None


def row(dump, stem, key, policy, budget, ref):
    """Success and Routes over all targets; the two conditional columns per target."""
    recs = _read(R / dump)
    if ONLY is not None:
        recs = [r for r in recs if r["target"] in ONLY]
    labeled = _read(L / f"labeled_{stem}.jsonl")
    stamp, per_rec = {}, {}
    for rec in recs:
        st = policy_stamps(rec) if policy else {
            route_key(x["steps"]): x.get("found_at_unique")
            for x in (rec.get(key) or [])}
        per_rec[rec["target"]] = st
        for k, v in st.items():
            stamp[(rec["target"], k)] = v
    best = {}
    for r in labeled:
        c = composite(r)
        fu = stamp.get((r["target"], route_key(r["steps"])))
        if c is not None and fu is not None and fu <= budget:
            best[r["target"]] = max(best.get(r["target"], 0.0), c)
    n, solved, uniq = len(recs), 0, []
    cut_by, btr_by = {}, {}
    for rec in recs:
        t = rec["target"]
        st = per_rec[t]
        kept = [x for x in (rec.get(key) or [])
                if (st.get(route_key(x["steps"])) or 10 ** 9) <= budget]
        solved += bool(kept)
        seen = {}
        for x in kept:
            seen.setdefault(leaves(x), x)
        u = list(seen.values())
        uniq.append(len(u))
        cuts = [c for c in (first_cut(x, t) for x in u) if c is not None]
        if len(cuts) >= 2:
            pp = list(combinations(cuts, 2))
            cut_by[t] = (statistics.mean(float(a != b) for a, b in pp), len(pp))
        if t in ref and t in best:
            btr_by[t] = float(best[t] > ref[t])
    btr_all = statistics.mean(float(best.get(rec["target"], 0.0) > ref[rec["target"]])
                              for rec in recs if rec["target"] in ref)
    m = lambda v: statistics.mean(v) if v else float("nan")
    return dict(Success=solved / n, Routes=m(uniq), _btr_all=btr_all,
                _cut=cut_by, _btr=btr_by)


COLS = [("Success", ".3f"), ("Routes", ".2f"), ("CutDiff", ".3f"), ("BTR", ".3f")]
HEAD = {"CutDiff": "1stCutDiff"}
MIN_PAIRS = [50]


def cell(v, f, tex=False):
    if v != v:                       # nan: suppressed for want of support
        return "--"
    return f"${v:{f}}$" if tex else f"{v:{f}}"


def delta(v, p, f, tex=False):
    """A delta is only printed where BOTH the row and its parent carry a value."""
    if v != v or p != p:
        return cell(v, f, tex)
    if tex:
        return f"${v:{f}}$\\,\\textcolor{{gray}}{{$({v-p:+{f}})$}}"
    return f"{v:{f}} ({v-p:+{f}})"


def aggregate(rows, common):
    """Turn each arm's per-target values into the printed number.

    Per arm (the default) every column is read on whatever targets that arm happens to
    reach, which is how the same statistic ends up resting on 13 targets for one row and
    52 for another. Under `common` each of the two conditional columns is read on the
    INTERSECTION over the arms in the table -- the targets where the column is defined
    for every one of them -- so the rows describe one population and can be subtracted
    from each other. Success and Routes are deliberately left on all 60: they ARE the
    coverage terms -- a target an arm never solved is a real zero for both -- and
    restricting them to the intersection would define the difference away. Success in
    particular is 1.000 for every arm on either intersection, by construction. Routes also
    has to stay visible next to 1stCutDiff, which is read against it.

    The two intersections are not the same set, because the two columns need different
    things: 1stCutDiff needs two routes to compare, BTR needs one scored route. Taking
    the stricter set for both would price BTR on the easy targets only.

    Under `common`, BTR no longer charges an arm for targets it failed -- those are out
    of the intersection -- so it reads as route QUALITY given a route, with coverage
    left to Success. The default keeps the old behaviour, where a miss scores 0.
    """
    if not common:
        for r in rows:
            own = r["_cut"]
            pairs = sum(n for _, n in own.values())
            r["CutDiff"] = (statistics.mean(v for v, _ in own.values())
                            if own and pairs >= MIN_PAIRS[0] else float("nan"))
            r["BTR"] = r["_btr_all"]
            r["_cut_n"], r["_pairs"], r["_btr_n"] = len(own), pairs, None
        return None, None
    cs = set.intersection(*[set(r["_cut"]) for r in rows]) if rows else set()
    bs = set.intersection(*[set(r["_btr"]) for r in rows]) if rows else set()
    for r in rows:
        pairs = sum(r["_cut"][t][1] for t in cs)
        r["CutDiff"] = (statistics.mean(r["_cut"][t][0] for t in cs)
                        if cs and pairs >= MIN_PAIRS[0] else float("nan"))
        r["BTR"] = statistics.mean(r["_btr"][t] for t in bs) if bs else float("nan")
        r["_cut_n"], r["_pairs"], r["_btr_n"] = len(cs), pairs, len(bs)
    return cs, bs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--common", action="store_true",
                    help="read 1stCutDiff and BTR on the targets where they are defined "
                         "for EVERY arm in the table; Success and Routes stay on all 60")
    ap.add_argument("--min-pairs", type=int, default=50,
                    help="print 1stCutDiff as -- below this many route pairs")
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    MIN_PAIRS[0] = a.min_pairs
    ref = reference()
    B = a.budget
    pop = "common targets" if a.common else "each arm's own targets"

    t1 = [(lab, row(dump, stem, key, pol, B, ref))
          for lab, dump, stem, key, pol in EXPANSION]
    c1, b1 = aggregate([v for _, v in t1], a.common)

    V = {c: row(f"v22_ladder_{c}_cf190/iter.jsonl", f"rq_v22_ladder_{c}_cf190",
                "routes_union", False, B, ref) for _, c, _ in LADDER}
    c2, b2 = aggregate([V[c] for _, c, _ in LADDER], a.common)

    print(f"TABLE 1  expansion policy      cut at {B} unique molecules opened"
          f"      1stCutDiff/BTR on {pop}\n")
    print(f"{'Method':<26}" + "".join(f"{HEAD.get(k, k):>12}" for k, _ in COLS))
    for lab, v in t1:
        print(f"{lab:<26}" + "".join(f"{cell(v[k], f):>12}" for k, f in COLS))

    print(f"\nTABLE 2  reasoning ladder      cut at {B}      "
          f"1stCutDiff/BTR on {pop}\n")
    print(f"{'Variant':<16}{'M D C dv or':<14}{'Par.':<8}"
          + "".join(f"{HEAD.get(k, k):>18}" for k, _ in COLS))
    for disp, c, par in LADDER:
        fl = " ".join("Y" if x else "-" for x in FLAGS[c])
        cells = [cell(V[c][k], f) if par is None
                 else delta(V[c][k], V[par][k], f) for k, f in COLS]
        print(f"{disp.replace(chr(92)+'text',''):<16}{fl:<14}"
              f"{PARENT_LABEL[c]:<8}" + "".join(f"{x:>18}" for x in cells))

    print("\nsupport")
    if a.common:
        print(f"  TABLE 1   1stCutDiff on {len(c1)} common targets, "
              f"BTR on {len(b1)} common targets")
        print(f"  TABLE 2   1stCutDiff on {len(c2)} common targets, "
              f"BTR on {len(b2)} common targets")
        print(f"  route pairs behind 1stCutDiff (-- below {a.min_pairs}):")
    else:
        print(f"  targets with >=2 routes, and route pairs "
              f"(-- below {a.min_pairs} pairs):")
    for lab, v in t1 + [(c, V[c]) for _, c, _ in LADDER]:
        print(f"    {lab:<26}{v['_cut_n']:>4} targets{v['_pairs']:>7} pairs")

    if UNRESOLVED[0]:
        print(f"\nnote: ard.first_cut returned None for {UNRESOLVED[0]}/{UNRESOLVED[1]} "
              f"routes (dropped from 1stCutDiff only) -- the exact-string target match "
              f"no longer holds on this dump", file=sys.stderr)

    if a.latex:
        print("\n% ---- Table 1")
        for lab, v in t1:
            print(f"{lab:<26}& " + " & ".join(cell(v[k], f, True) for k, f in COLS)
                  + r" \\")
        print("\n% ---- Table 2")
        for disp, c, par in LADDER:
            fl = " & ".join(r"\checkmark" if x else "--" for x in FLAGS[c])
            cells = [cell(V[c][k], f, True) if par is None
                     else delta(V[c][k], V[par][k], f, True) for k, f in COLS]
            print(f"{disp} & {fl} & {PARENT_LABEL[c]} & " + " & ".join(cells) + r" \\")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
