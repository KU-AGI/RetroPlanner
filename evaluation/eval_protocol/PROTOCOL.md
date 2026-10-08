# Evaluation protocol

Every planner is evaluated in the same environment: the same single-step model, the
same stock, the same budget unit, and the same scoring code. This file defines that
environment, how each baseline consumes the budget, and the metrics.

## Setting

| | |
|---|---|
| benchmarks | USPTO-190 (in-distribution; `targets_uspto190.jsonl`, which carries the published reference routes) and ChEMBL-1000 (out-of-distribution; no reference routes) |
| single-step model | R-SMILES, top-K = 10 candidates per molecule, served live. Every candidate any planner sees comes from this model, so a row measures the planner, not its expansion policy |
| stock | eMolecules (`--stock emols`, full InChIKey). A route is solved only when all its leaves are in stock |
| budget N_B | unique single-step calls per target. Main results use N_B = 50; the ablations use N_B = 300 |

### The budget unit

The budget counts **unique** single-step calls: expanding a molecule that was already
expanded for this target costs nothing, including across restarts. This is the unit
RetroAgent and Retro-R1 count, so it is the one every method shares. Each route is
stamped with the budget spent when it was first found, and a method's result at N_B
is the set of routes stamped at or below N_B.

### How each method consumes the budget

- **Search algorithms** (MCTS, Retro\*-0, Retro\*, MORetro\*) run once per budget cell
  under the common expansion budget. Node selection uses an equal-weight scalarization
  of plausibility, round-trip consistency and starting-material price in place of the
  single-step confidence; MORetro\* keeps its Bayesian-optimization sampling over
  objective weights.
- **Retro-R1** searches within a single continuous rollout over multiple rounds.
  Independent runs are laid end to end on the cumulative budget axis, so run *i*
  starts where run *i-1* stopped.
- **RetroAgent** restarts failed rollouts until a route is found. Its runs are laid
  end to end in the same way, each charged the budget it actually spent
  (`runners/build_spend.py`), including runs that found no route.
- **RetroPlanner** ends each rollout at its first complete route and restarts,
  spending the remaining budget on alternative routes, up to 40 rollouts. It does not
  shuffle the environment: it samples its reasoning and decisions at temperature 0.6
  with a per-rollout seed, so variation across rollouts comes from the agent itself.
  The run goes to a ceiling of 500 unique calls, and each cell is read off the
  route stamps (`runners/retroplanner_rsmiles.sh`).

## Metrics

All metrics are computed over the full benchmark T. R_T is the set of routes a method
returns for target T within N_B whose leaves all lie in stock; routes reaching the
same leaf set count as one. S = {T : R_T non-empty} is the set of solved targets.

Per-reaction signals:

- **p(r)** — reaction plausibility from the AiZynthFinder filter policy; plausible
  when p >= 0.05.
- **rt(r)** — round-trip consistency: the rank at which the ReactionT5v2 forward model
  recovers the product from the proposed precursors; consistent when rt <= 5.
- **c(m)** — MolPrice predicted price of a molecule, USD/mmol.

Per-route quantities, for a route ρ with reactions r and leaves leaf(ρ):

- u_p(ρ) = fraction of reactions with p(r) >= 0.05
- u_rt(ρ) = fraction of reactions with rt(r) <= 5
- ũ_c(ρ) = sum of c(m) over leaf(ρ)
- ū_c(ρ) = log(1 + min(ũ_c(ρ), P_max)) / log(1 + P_max), with P_max = 1000, and
  u_c(ρ) = 1 − ū_c(ρ), so all three objectives are higher-is-better
- s(ρ) = (u_p(ρ) + u_rt(ρ) + u_c(ρ)) / 3

**Representative route.** Route quality is evaluated on one composite-best route per
solved target, ρ*_T = argmax over R_T of s(ρ). The price scale is fixed by P_max alone,
so a route gets the same score whichever planner returns it.

| metric | definition |
|---|---|
| Success Rate | \|S\| / \|T\| |
| Routes | number of distinct routes returned per solved target |
| Plausibility | (1/\|T\|) · sum over T in S of u_p(ρ*_T); unsolved targets score 0 |
| Round-Trip | (1/\|T\|) · sum over T in S of u_rt(ρ*_T); unsolved targets score 0 |
| Price | geometric mean over all of T of min(ũ_c(ρ*_T), P_max), with each unsolved target scored at P_max |
| Better-Than-Ref | fraction of targets whose representative route scores higher than the target's published reference route over the same three objectives. An unsolved target, or a tie with the reference, counts as a loss. USPTO-190 only |

Unsolved targets are not dropped: they score 0 on plausibility and round-trip and
P_max on price. The same cap clips solved routes, so failing a target never scores
better than solving it. `metrics/table_geo.py` also prints the same columns over
solved targets only (`IMPUTE=0`); the two answer different questions.

## Running it

1. `runners/retroplanner_rsmiles.sh` — serves the checkpoint and runs RetroPlanner on
   USPTO-190 with the setting above (developer message:
   `developer/dev_retroplanner.txt`).
2. `runners/build_spend.py` — builds the per-run spend file that lays the Retro-R1
   and RetroAgent runs end to end.
3. `runners/table_main.sh {uspto|chembl|both}` — renders the main table at
   N_B = 50 and 300 through `metrics/table_geo.py`.

The board harness itself is `evaluation/board/` (`eval_board_agent.py` and its
forced-reasoning and iterative-restart drivers).
