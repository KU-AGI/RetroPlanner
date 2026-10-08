#!/usr/bin/env python
"""The iterative-restart protocol, run with the analysis channel FORCED open.

This is how RetroPlanner consumes the cumulative budget: each rollout ends at its
first complete route (`--stop-episode-on-solve`) and a fresh rollout starts on a
fresh board, spending the remaining budget on alternative routes. The environment
is not shuffled; restarts differ because the agent samples its reasoning and
decisions (`--temperature 0.6`, per-rollout seed). Each episode is driven by
`eval_board_agent_forced_reasoning.py`, which renders the exact training-time
prefix and injects `<|channel|>analysis<|message|>`, so the reasoning arm is
evaluated with its reasoning on; this wrapper adds the restart loop and the
budget accounting.

The budget is counted in unique single-step calls: DISTINCT opened SMILES per
target, accumulated across restarts, so a molecule an earlier episode already
expanded is free on a restart. Every route is stamped with the unique-call count
at which it was first found, so success and route quality at any budget up to the
ceiling are read off one run.

    ITER_MAX_BUDGET=500 ITER_MAX_ROLLOUTS=40 ITER_STOP_ON_SOLVE=0 ITER_PATIENCE=0 \\
    python "$RP_BOARD"/eval_board_agent_forced_iterative.py \\
        --targets "$RP_PROTOCOL"/targets_uspto190.jsonl \\
        --model-url http://127.0.0.1:$RP_PORT_LLM/v1 --model retroplanner \\
        --menu-url http://127.0.0.1:$RP_PORT_MENU/predict --stock emols \\
        --rt live:http://127.0.0.1:$RP_PORT_FORWARD \\
        --developer-file "$RP_PROTOCOL"/developer/dev_retroplanner.txt --reasoning medium \\
        --two-stage --ctx-len 131072 --budget 300 --max-turns 120 --max-tokens 3000 \\
        --illegal-cap 5 --stop-episode-on-solve --temperature 0.6 \\
        --workers $RP_WORKERS --out results/forced_iter_uspto190.jsonl

`runners/retroplanner_rsmiles.sh` runs exactly this, sharded over replicas.

Env knobs:
    ITER_MAX_BUDGET    unique single-step calls per target   (default 500)
    ITER_MAX_ROLLOUTS  hard cap on episodes per target       (default 50)
    ITER_PATIENCE      stop after N episodes that opened nothing new (default 3; 0 = off)
    ITER_STOP_ON_SOLVE end the target at its first solve     (default 1)
    ITER_MAX_CALLS     cap on TOTAL open calls, revisits included (default 0 = off)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

SD = Path(__file__).resolve().parent
sys.path.insert(0, str(SD))

import eval_board_agent_forced_reasoning as base     # noqa: E402
from board.state import Board                        # noqa: E402

MAX_BUDGET = int(os.environ.get("ITER_MAX_BUDGET", "500"))
MAX_ROLLOUTS = int(os.environ.get("ITER_MAX_ROLLOUTS", "50"))
PATIENCE = int(os.environ.get("ITER_PATIENCE", "3"))
MAX_CALLS = int(os.environ.get("ITER_MAX_CALLS", "0"))
"""Stop a target when the TOTAL open calls across its episodes reach this. 0 disables it.

Why a second call budget. `ITER_MAX_BUDGET` counts DISTINCT molecules -- an episode that
re-opens what an earlier one opened pays nothing against it. This one counts every open,
revisits included, for a comparison on total expansions. Applying it at run time rather than
by replaying a recording afterwards is the only way the route set that survives the cap is
actually known -- routes are not attributable to episodes after the fact. The reported
protocol leaves it off, so the unique budget is the one that binds."""
RESUME = os.environ.get("ITER_RESUME", "")
"""Continue a finished run instead of starting over, on the TOTAL-CALL axis only.

Episodes are independent restarts against a fresh board -- nothing but accounting crosses
between them -- so "run to 500" and "run to 300, then run 200 more" draw from the same
distribution. What the prior dump carries is `calls_total`, `n_episodes` and the stamped
`routes_union`; the loop resumes the call counter, CONTINUES the episode index so
`seed = seed_base + idx` does not replay the earlier episodes (a repeated seed reproduces the
trajectory and the extra budget buys nothing), and keeps every earlier route with its original
stamp.

WHAT CANNOT BE RESUMED. `seen` -- the set of opened SMILES -- is not in the dump, only its
size. So the unique-molecule budget restarts from zero and `found_at_unique` is left None on
routes found after the resume; a resumed dump is valid on the TOTAL axis and must not be used
for the unique one.
"""
STOP_ON_SOLVE = os.environ.get("ITER_STOP_ON_SOLVE", "1") == "1"
"""Whether the restart loop ends at the first solve.

RetroAgent's protocol says it does -- "until the search budget is exhausted OR a route
is found" -- and that is the right rule for a solve rate and for the
`solved_at_budget` curve, so it stays the default.

It is the wrong rule for RetroPlanner's protocol, which spends the remaining budget on
alternative routes: with it on, a run stops at its first route and never spends the
budget it was given. Set ITER_STOP_ON_SOLVE=0 to keep restarting until the budget or
patience runs out, and read `routes_union` instead of `routes`."""


_tls = threading.local()


class TrackingBoard(Board):
    """Board that records which SMILES it actually opened.

    Only successful opens are recorded: `_do_open` raises on an illegal one, so
    appending after the super() call keeps rejected actions out of the budget --
    a rejected action never reached the single-step server and never cost one.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opened_smiles: list[str] = []
        _tls.last_board = self

    def _do_open(self, act) -> None:
        super()._do_open(act)
        self.opened_smiles.append(self.mols[act.mid].smiles)


base.Board = TrackingBoard
_PRIOR: dict[str, dict] = {}
if RESUME:
    for _l in open(RESUME):
        if _l.strip():
            _r = json.loads(_l)
            _PRIOR[_r["target"]] = _r
    print(f"# resume: {len(_PRIOR)} targets from {RESUME}", file=sys.stderr)


_episode = base.run_target


def _forced_iterative_run_target(target, world, client, style, **kw):
    """One EXPERIMENT for one target: forced-reasoning episodes until budget,
    solve, or patience."""
    t0 = time.time()
    # The base dispatcher passes `rollout=` down. This wrapper supplies its own per-episode index,
    # so the caller's value would collide -- drop it here and keep it only for the record.
    outer_rollout = kw.pop("rollout", 0)
    seen: set[str] = set()
    budget = 0
    calls_total = 0
    zero_streak = 0
    episodes: list[dict] = []
    decisive: dict | None = None
    solved_at_budget = None
    prior = _PRIOR.get(target) or {}
    ep_offset = int(prior.get("n_episodes") or 0)
    calls_total = int(prior.get("calls_total") or 0)
    solved_at_budget = prior.get("solved_at_budget")
    # Routes are unioned across restarts, keyed by their step set, because two episodes
    # that find the same route found one route. Without the key a rerun of the same
    # trajectory would inflate the count -- and at temperature 0 a rerun is the common case.
    routes_union: dict[tuple, dict] = {}
    for _rt in (prior.get("routes_union") or []):
        _k = tuple(sorted((str(p), tuple(sorted(map(str, rr))))
                          for p, rr in (_rt.get("steps") or [])))
        if _k:
            routes_union[_k] = _rt

    exit_reason = "max_rollouts"
    for step_i in range(MAX_ROLLOUTS):
        idx = ep_offset + step_i          # seeds continue past the resumed run's episodes
        out = _episode(target, world, client, style, rollout=idx, **kw)
        board = getattr(_tls, "last_board", None)
        opened = list(getattr(board, "opened_smiles", []))
        new = [s for s in opened if s not in seen]
        seen.update(new)
        budget += len(new)
        calls_total += out.get("calls") or 0

        n_out = out.get("output") or []
        episodes.append({
            "i": idx, "solved": out.get("solved"), "stop": out.get("stop"),
            "turns": out.get("turns"), "calls": out.get("calls"),
            "new_calls": len(new), "cum_budget": budget,
            "n_routes": len(out.get("routes") or []),
            # how much of this episode actually reasoned -- the thing the whole
            # forced path exists to produce, kept per episode so a run can be
            # read for "did it stop reasoning as it went" and not just in total
            "n_turns_reasoned": sum(1 for o in n_out if o.get("reasoning")),
            "n_turns_out": len(n_out),
        })
        for rt in (out.get("routes") or []):
            key = tuple(sorted(
                (str(p), tuple(sorted(map(str, rr)))) for p, rr in (rt.get("steps") or [])))
            if key and key not in routes_union:
                # DISCOVERY BUDGET, stamped at first sight. Without this the dump cannot say
                # when a route appeared: `episodes[].n_routes` is a count, so a post-hoc replay
                # at a call cap knows how many routes the surviving episodes produced but not
                # WHICH, and a Pareto front over the surviving set is then uncomputable. Both
                # units are recorded because the arms are budgeted on different ones -- the
                # search arms on total expansions, RetroAgent and Retro-R1 on distinct.
                rt["found_at_calls"] = calls_total
                # `seen` did not survive the resume, so the unique counter restarted and this
                # figure would understate the true cumulative distinct count -- record None
                # rather than a number that looks usable.
                rt["found_at_unique"] = None if RESUME else budget
                rt["found_at_episode"] = idx
                routes_union[key] = rt
        if decisive is None or out.get("solved"):
            decisive = out
        if out.get("solved"):
            if solved_at_budget is None:
                solved_at_budget = budget
            if STOP_ON_SOLVE:
                exit_reason = "solved"
                break
        if out.get("stop") in ("api_permanent", "crash"):
            exit_reason = out.get("stop")
            break
        if len(new) == 0:
            zero_streak += 1
            if PATIENCE and zero_streak >= PATIENCE:
                exit_reason = "patience"
                break
        else:
            zero_streak = 0
        if budget >= MAX_BUDGET:
            exit_reason = "budget"
            break
        if MAX_CALLS and calls_total >= MAX_CALLS:
            exit_reason = "calls"
            break

    rec = dict(decisive or {})
    rec.update({
        "target": target,
        "rollout": outer_rollout,
        "solved": bool(solved_at_budget is not None),
        "solved_at_budget": solved_at_budget,
        "budget_used_unique": budget,
        "calls_total": calls_total,
        "n_episodes": ep_offset + len(episodes),
        "n_episodes_this_run": len(episodes),
        "resumed_from": RESUME or None,
        "episodes": episodes,
        "routes_union": list(routes_union.values()),
        "n_routes_union": len(routes_union),
        "stop_on_solve": STOP_ON_SOLVE,
        "iter_wall_s": round(time.time() - t0, 2),
        # why the RESTART LOOP ended -- not whether the target was solved. In
        # diversity mode the loop runs on past a solve, so folding the two together
        # would report "solved" for runs that actually died on patience and hide
        # whether a run reached the budget.
        "iter_stop": exit_reason,
    })
    return rec


base.run_target = _forced_iterative_run_target


if __name__ == "__main__":
    print(f"# forced + iterative: max_budget={MAX_BUDGET} (unique single-step calls per "
          f"target) max_calls={MAX_CALLS or 'off'} (TOTAL open calls) "
          f"max_rollouts={MAX_ROLLOUTS} patience={PATIENCE} "
          f"stop_on_solve={STOP_ON_SOLVE}", file=sys.stderr)
    if not STOP_ON_SOLVE:
        print("# DIVERSITY MODE: restarts continue past the first solve until the budget "
              "or patience runs out; read routes_union / n_routes_union", file=sys.stderr)
    raise SystemExit(base.main())
