# board — the planning environment

The environment RetroPlanner plans in, and the code that turns a recorded search into
training episodes for it:

* the search graph `G_t` as an AND–OR DAG, and the three stages of a turn — **open**,
  **rank**, **done** (paper Sections 2.1–2.2);
* the textual rendering of that graph the planner reads, with the per-candidate signals
  `q`, `p`, `rt` and `$` (Appendix A.4, "Environments");
* the gpt-oss Harmony encoding of an episode;
* the chemical evidence a reasoning trace may cite, and the fact-check the traces are
  verified against (Section 2.3, Appendix A.3, "Chemical Evidence").

```
state.py      molecules, reactions, and what each action does to them.  No text.
render.py     Board -> the text the model reads.  Every formatting decision lives here.
parse.py      text / tool-call JSON -> actions, and the rendered board -> a state view,
              used to prove the rendering is lossless.
episode.py    the recorded search joined into an episode: the DP that labels actions,
              breadth-first replay of the selected routes, and the per-turn evidence.
augment.py    several training instances per target from the same recorded routes.
evidence.py   bond changes, earned reaction names, molecule descriptors.
factcheck.py  does a drafted reasoning trace agree with the facts of its turn?
harmony.py    the episode as Harmony messages; the developer instructions and tool schema.
```

Consumers: `../eval_board_agent.py` (the live environment used in evaluation),
`tools/reaction-mcp/scripts/traj_route_reasoning_board.py` (teacher reasoning traces),
`tools/reaction-mcp/scripts/verify_harmony.py` (token-level rendering) and
`torchtitan/torchtitan/experiments/retroplanner/` (SFT).

Self-test — render a hand-built board, parse it back, and move it with parsed actions:

```
. config/env.sh
cd evaluation/board && "$(rp_py "$RP_ENV_BOARD")" -m board.parse --selftest
```

## The graph and the actions

A molecule is an OR node: it is solved as soon as any one reaction under it is solved. A
reaction is an AND node: it is solved only when every precursor is. A molecule in stock is
solved on arrival. A molecule reached along two routes is one node with one id, so the
routes the planner has explored share one graph and a candidate can be compared against
them (`Board.add_molecule`).

| action | effect | charged |
|---|---|---|
| `open(mid)` | fetch the K = 10 candidate disconnections of a frontier molecule | 1 single-step call |
| `rank(mid, order, take)` | order the candidates worth taking and apply the first `take` of them now; the rest stay on the board to claim later | 0 |
| `done(choices)` | claim a route: one candidate number per molecule it makes. Reactions the claim needs are created; the claim is refused whole if any leaf is not purchasable | 0 |
| hand-over | the final message; names the claimed routes and ends the episode | — |

`take > 1` and opening several molecules in one call are what make the expansion
multi-route: several subtrees exist at the same depth at once, and the next rank turn
compares rival cuts instead of committing to one branch. The budget counts `open` only,
i.e. single-step calls (`state.COST`).

The board rejects, with a message the model can act on: an action on an unknown or dead
molecule, a candidate not on the menu or already tried, two actions on one molecule in one
call, and any candidate that would make a molecule its own precursor. `dead` (with three
named reasons) and `analyze` (buy the evidence for named candidates) are implemented in
`state.py` but are not in the tool schema the model is given (`harmony.ACT_SCHEMA`).

## The board

Everything below is a field on `render.RenderStyle`, or an environment flag.

| part | example | knob |
|---|---|---|
| event, rank | `[Event] e3 ranked [c1] · c1 applied → r1 · pieces:` then one line per piece with the status it arrives in, then the cascade | `indent_event`, `indent_event_tail` |
| event, claim | `[Event] ROUTE CLAIMED · A = r1 · r3` | — |
| routes | one line per claimed route: steps, `p N/M pass`, `rt N/M back`, leaves, `$` | `routes_detail`, `BOARD_ROUTE_AXES`, `BOARD_ROUTES_FULL` |
| ledger | `r1  e3·c1  1 of 2 closed        e3 ranked: c1   (was —)` | `ledger_gap`, `show_ranked_column`, `show_was_on_first` |
| frontier | `  n4   under r1 · depth 1 · no candidates yet` | `indent_open`, `id_col` |
| candidate | `<c0 q.374 p.951 rt1 $.604>SMILES*($.128) + SMILES($.410)</c0>` | `signal_order`, `decimals`, `drop_leading_zero`, `price_fmt` |
| menu tail | `... 5 more, best q .004` | `menu_show`, `menu_cutoff`, `cutoff_signal` |
| closed | `CLOSED  k2*($.128) · d7*($.301)` — the running bill of materials | `price_in_closed` |
| molecule | `<mol e3>SMILES</mol>` once, `<mol e3/>` after | `Board.shown` |

The signals on a candidate line:

* `q` — the single-step model's confidence.
* `p` — reaction plausibility from the AiZynthFinder filter; a step is plausible when
  `p ≥ 0.05`.
* `rt` — the rank at which the ReactionT5v2 forward model recovers the opened molecule from
  the precursors. Three states are rendered apart: `rt1` (recovered at that rank), `rt✗`
  (the forward model ran and did not recover it), `rt?` (no entry, nothing was asked).
* `$` — with `BOARD_COST_NORM=1`, the log-sum-exp of the MolPrice predictions of the
  precursors, normalised to [0, 1]; lower is cheaper. Each precursor's own price follows its
  SMILES, and `*` marks a precursor in stock. Without the flag, prices print in USD/mmol.

A score is written without its leading zero (`q.374`), so a bare decimal on the board is a
score and an integer is a count. The depth cap and the call budget are not printed: they are
properties of how an episode is bounded, not facts to reason from (`RenderStyle.show_budget`).

### Environment flags

| flag | effect |
|---|---|
| `BOARD_COST_NORM=1` | `$` on the normalised 0–1 scale (`render.dollars`) |
| `BOARD_ROUTE_AXES=1` | the ROUTES line carries the three route axes |
| `BOARD_ROUTES_FULL=1` | every claim turn lists every claimed route, not only the new ones |
| `BOARD_ROUTE_FRONT=1` | the ROUTES block also marks its own Pareto front (`▲`, `≺C`) |
| `BOARD_MENU_FILTER=1` | candidates that would close a cycle are never put on screen |
| `BOARD_MENU_DEDUP_RXN=1` | with the filter, a (product, precursors) pair already in the graph is not shown again |
| `BOARD_REQUIRE_SCORES=1` | raise if a rendered candidate carries a signal no cache scored (corpus building) |
| `BOARD_PRICE_NOTE=1` | add the "taken cut is also the cheapest" note to the teacher-private evidence |

The evaluation runner (`evaluation/eval_protocol/runners/retroplanner_rsmiles.sh`) serves
`BOARD_COST_NORM`, `BOARD_ROUTE_AXES`, `BOARD_ROUTES_FULL`, `BOARD_MENU_FILTER` and
`BOARD_MENU_DEDUP_RXN`, and unsets the rest. The developer message and the tool schema are
generated from the same flags and the same `RenderStyle`, so the instructions never describe
a mark the board does not print.

## From recorded search to episode

The inputs are the Retro\*-0 search records on PaRoutes targets: the cached top-10 menus
(`data/route_search/draw_cache`), the search graph with per-candidate `used / dead /
unexpanded` labels, and the selected routes (the first two Pareto fronts, up to eight routes
per target). `episode.py` joins them; no tool is re-run.

* **Replay.** Every selected route of a target is realised in one episode
  (`Policy.route_queue`). With `branch_queue`, each rank commits exactly the candidates some
  queued route still needs at that molecule, so the merged route DAG is replayed
  breadth-first: the same reactions, laid out in space rather than in time. Routes the graph
  can already make are claimed with `done` in one action.
* **Labels.** Where no route dictates the step, actions come from a DP over the menus:
  `value(mol) = max over candidates of min(p, min over precursors value(precursor))`, with a
  stock molecule unconstrained and a molecule past the depth bound `-inf`. Ties within
  `tie_eps` are broken on `rt`, then price, then `q`. A route whose own step is not on the
  top-10 menu raises `OffRoute` and the instance is dropped.
* **Mistakes.** Optionally the rollout departs from the label once (`Policy.mistakes`), walks
  into a candidate that does not close, and recovers. Those turns are marked
  `supervised=False` and carry no loss. A molecule the recording never expanded raises
  `NoMenu` and the episode is dropped rather than labelled.
* **Variants.** `augment.py` builds several instances per target from the same routes:
  renumbered menus, a different claiming order, route clusters, single routes, a trap, and
  free claims.

Each turn carries `evidence` (`episode.evidence_for`): the numbers and facts a reasoning trace
for that turn may cite. It includes teacher-private facts — the bond changes, earned reaction
names and split shape of the candidates being ranked, and which axis actually separated the
taken candidate (`decided_by`) — so the teacher can explain an action it did not choose
without inventing the reason.

## Chemical evidence and the fact-check

`evidence.py` supplies, per (product, precursors): the formed, broken and order-changed bonds
from RXNMapper atom mapping, the reaction name and whether a named template applies and
reproduces the precursors, and RDKit descriptors (scaffold, rings, heavy atoms, stereocentres)
with the product-to-precursor split shape. `CacheEvidence` reads the precomputed caches;
`LiveEvidence` calls the `reaction-mcp` tools for states the caches do not hold. A failed
mapping is reported as not measured, never as "no bond changed".

`factcheck.check(text, evidence, kind, ...)` refutes a drafted trace only where a claim
contradicts a fact present in the turn's evidence: a quoted number not on the board, an
argmax or cut-off claim that is false, an unearned reaction name, a ring or functional-group
word no molecule on the turn matches, a purchasability or completeness claim the state
refutes, an `rt` claim against the rt column, prose that argues for a candidate the call does
not take, and recycled sentences. Every check fails open when the fact is absent.
`describe()` turns the violations into the correction the teacher is re-prompted with.

## Harmony encoding

`harmony.episode_messages` emits an episode as structured messages (role, channel, recipient,
content type, content), never as token text; the token layer is rendered by
`tools/reaction-mcp/scripts/verify_harmony.py` exactly as vLLM renders it at inference.

```
developer                 task, board semantics, notation, the signals, the reasoning form
user                      the opening board
assistant -> board_act    {"actions": [...]}        commentary, json
functions.board_act       the board that results    commentary
...
assistant                 the hand-over             final
```

* `done="final"` (default): the stop decision is the final message, not a tool call, so no
  episode ends on a call that expects a result.
* `first_board="user"` (default): the opening board has no call before it, so it is a user
  message rather than a tool result.
* `analysis`: `none` for the no-reasoning board, `text` to emit each turn's teacher trace on
  the analysis channel.

`to_chat_messages` gives the same episode in the shape gpt-oss's `chat_template.jinja`
expects. That template serialises tool results through `tojson`, which escapes the board's
`<c0 ...>` tags and newlines, so the board it produces differs from the one vLLM serves;
train on the Harmony rendering, or patch that branch of the template.
