# Training-data construction

The training corpus is built from the PaRoutes training split in four stages: search each
target, select a diverse set of high-ranking routes, replay them as board episodes, and write
the teacher reasoning for every turn.

    traj_route_search.py        targets  -> search runs   (Retro*-0 + R-SMILES, search graph)
    traj_route_pool.py          runs     -> pooled routes (deduplicated)
    traj_route_score.py         routes   -> node caches   (plausibility, round-trip, price)
    traj_route_label.py         routes   -> sft_candidates_<ds>.jsonl (Pareto selection)
    board/episode.py            routes   -> board episodes (merge, breadth-first replay)
    run_reasoning.sh            episodes -> reasoning     (teacher traces, verified)
    verify_harmony.py --augment reasoning -> wire rows    (the text vLLM renders)
    torchtitan ...prepare_board_dataset
                                wire rows -> training sequences

`traj_route_common.py` holds what every stage must agree on: the reaction key shared with the
evaluation caches, the route keys, the stock predicate, and the single-step client.

## 1. Search

Each target is searched once with Retro*-0 (the single-step model's own `-log p` as the
AND-node cost, no learned cost-to-go), using R-SMILES as a template-free single-step model, under
a budget of 300 unique single-step calls:

    python traj_route_search.py --ds paroutes --models rsmiles --algos retrostar \
        --budget 300 --draws 1 --dump-graph

`--dump-graph` records the search graph per target: every molecule opened, whether it reached
purchasable leaves, and each top-k candidate labelled `used`, `dead` or `unexpanded`. The
recorded routes carry only what succeeded; the graph is the only source of dead ends, and it is
what the reasoning is later verified against.

Menus are frozen per search and written to a draw cache keyed by (model, draw, top_k), so a
replay sees exactly the chemistry the search saw. A parallel search gives each shard its own
writable cache (`--draw-cache`) on top of read-only shared ones (`--draw-cache-base`), and
`merge_draw_cache.py` folds the shards back afterwards.

## 2. Scoring and selection

`traj_route_pool.py` pools every run into one table of distinct routes per target.
`traj_route_score.py` fills the per-node caches with the same implementations evaluation uses:
AiZynthFinder's filter policy for reaction plausibility (cut-off 0.05), ReactionT5v2 round-trip
recovery, and MolPrice for starting-material price.

`traj_route_label.py` aggregates them per route and ranks the routes of each target on the three
synthesis objectives: plausibility pass fraction, top-1 round-trip recovery, and starting-material
price. Top-1 recovery is a stricter criterion than the top-5 used in evaluation.

    python traj_route_label.py --stem paroutes --objectives feas_cost \
        --diverse --pool-rank 2 --select 8

The first two Pareto fronts form the candidate pool. The first front alone tends to concentrate
on routes with similar inexpensive starting materials; the second broadens route diversity
without substantially relaxing quality. From the pool, up to eight routes per target are kept,
walking it in objective order and accepting a route only if its starting-material set is new --
new by InChIKey skeleton and not a Tanimoto near-duplicate of one already kept, so a changed
leaving group or double-bond geometry does not count as a different route. The result is
`sft_candidates_<ds>.jsonl`.

## 3. Episodes

The selected routes of a target are merged at their shared intermediates into a single directed
acyclic graph, which is replayed breadth-first in a deterministic simulation of the planning
environment (`evaluation/board/board/episode.py`). Every observation is rebuilt from the recorded
menus, the search graph and the node caches, so the training observation is the one the
environment produces at inference. Each step becomes one of the planner's actions -- `open`,
`rank` with the number of committed candidates, or `done` -- and the episode ends with the
`final` hand-over.

Each turn also carries a snapshot of its evidence: the objective values of every candidate, and
the chemical evidence for each reaction -- bond changes from RXNMapper atom mappings represented
with CGRtools, the affected atoms and functional groups from RDKit, reaction classes and names
from Rxn-INSIGHT, template applicability, and molecular descriptors. A reaction whose mapping
fails is marked not measured, never as no bond change.

The episode text is fixed when it is rendered. A cache filled afterwards changes nothing until
the episodes are rendered again.

## 4. Reasoning

    ./run_reasoning.sh <side> <ncard>

drives `traj_route_reasoning_routeloop.py` over one input shard per teacher card, with
Qwen3.8-27B as the teacher. The teacher is shown each turn's board, its supporting evidence and
the action the replay took, and writes the reasoning that leads to that action.

**A chain, not independent calls.** Turn k is written with the earlier turns' reasoning and
decisions in context and asked to continue, so an episode reads as one line of thought: it refers
back to what it set aside, says which branch it is on, and does not re-derive what it already
established. Calls are sequential within an episode and parallel across episodes.

**One brief per action.** Open, rank, done and the hand-over are different questions and get
different briefs and different information:

* An `open` turn states that the piece cannot be bought and which pieces are already above it.
  Where the call opens several pieces it says why they are opened together -- rival cuts of one
  parent held at one depth, or co-precursors one reaction owes -- reading the relation off the
  board. Where it passes over a piece it could have opened, it says why. It names no chemistry:
  nothing structural is measured for a piece that has not been opened.
* A `rank` turn is written to a form (`traj_route_reasoning_routeloop.py`): the molecule, its
  disconnections in the board's own order, then the decision as a loop over the routes it
  declares. Each route says what it diverges from -- a ledger step of the same kind where one
  exists, otherwise the sibling candidates -- and which measure orders it. The disconnections
  are listed in board order, not rank order, because the reasoning is generated before the tool
  call: a description ordered by the answer means the answer was already decided.
* A `done` turn states that every leaf of the route is purchasable and which pieces are made
  rather than bought.
* The hand-over says what was built, why the routes are in this order, and which routes on the
  board are beaten on all three objectives.

**Verified against the board and the search graph.** Every draft passes two gates. `verify`
checks register: no oracle answer or label, no interface vocabulary, no atom-map index, no depth
ceiling, no post-hoc numbers mid-episode, and no rank-form headers on prose turns. `factcheck`
(`board/factcheck.py`) checks that the claims are true of the turn: a `q` appeal names the
candidate `q` actually favours, a below-cut-off candidate is not taken as sound, a named reaction
is one a template reproduces, a ring or group word matches a molecule on the turn, an unpriced
fragment is not called purchasable. For the rank form the slots make these checks exact: the
declared order is compared with the call's order, a reaction class with the template tier that
earned it, a bond claim with whether the mapping measured anything.

**Repaired rather than resampled.** A factually wrong draft is handed back with the fact that
refutes it and redrawn; a false claim is a wrong belief, and resampling the same prompt returns
the same sentence. Past that, draws continue at a rising temperature up to a cap, the least-bad
draft is kept, and a draft whose only defect is register is salvaged by substituting chemistry
for the interface words. A draft with a leak or a false claim is never salvaged. A turn left
empty is refilled later with `--fill`, since every later turn is written with it in context.

**Teacher fleet.** Each episode is pinned to one teacher replica. Turn k's prompt is turn k-1's
plus one observation and one thought, so nearly every prompt is a prefix that replica has already
seen; pinning keeps that prefix where it is cached. A replica that fails repeatedly is stepped
over until a cooldown expires, and its displaced episodes spread over the remaining replicas.

## 5. Checking what came out

    python traj_route_reasoning_routeloop.py --in <eps> --selftest

builds, from a real turn, one draft per gate broken in exactly one way, and fails if a gate does
not fire. Run it after touching any check. `factcheck` runs its schema checks inside one
exception guard; an exception there is recorded in `failed_checks` rather than failing the draft,
so a gate that reports nothing in the selftest should be looked for there first.

`verify_harmony.py` renders every episode through `openai_harmony`, checks that each tool call
parses back to the same actions and that no message body contains a Harmony special token, and
with `--augment` writes the preamble and assistant token spans exactly as vLLM renders them. That
augmented file is the training input.
