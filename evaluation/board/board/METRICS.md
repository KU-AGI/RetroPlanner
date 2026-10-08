# Metrics in the board pipeline, and where each comes from

Four stages produce numbers, and they answer different questions. Keep each number with
its own denominator.

| stage | file | what it can see |
|---|---|---|
| corpus check | `torchtitan/.../retroplanner/scripts/prepare_board_dataset.py` | the built episodes, before training |
| packing | `torchtitan/.../retroplanner/dataset.py` (`PackingStats`) | what the packer kept and dropped |
| training | `torchtitan/.../retroplanner/loss.py` + `board_metrics.py` | teacher-forced logits |
| evaluation | `evaluation/board/eval_board_agent.py` + `evaluation/eval_protocol/metrics/table_geo.py` | the agent playing live |

## 1 · Corpus check

`prepare_board_dataset.py --input <wire.jsonl>`; the per-record counts come from
`board_trajectory.validate_board_record`. Every `check/*` key except `n_*` counts a failure,
and each one is silent during training if left unchecked.

| metric | definition |
|---|---|
| `check/n_records`, `n_turns`, `n_calls`, `n_items` | corpus size: episodes, turns, `board_act` calls, actions inside them |
| `check/n_unsupervised` | turns produced by the rollout rather than the labeller (masked in the loss) |
| `check/action/unparsed` | a call that is not a legal `board_act` payload |
| `check/state/mid_not_open` | a call naming a molecule the board cannot act on (neither in OPEN nor in the ledger) |
| `check/state/candidate_off_screen` | a ranking naming a candidate past the end of the menu shown |
| `check/action/two_actions_one_molecule` | two actions on one molecule in one call |
| `check/final/no_route` | a hand-over with no ROUTES block |
| `length/{median,p95,max}` | episode length in tokens, through the real tokenizer |
| `length/dropped_at_seq_len` | episodes longer than the packing sequence length, which the packer drops whole |
| `data/routes_per_target_mean` | how many of a target's selected routes made it into the corpus |

## 2 · Packing

`PackingStats.as_dict()`, reported once per epoch.

| metric | definition |
|---|---|
| `data/trajectories_packed` / `_dropped` / `dropped_frac` | a dropped episode is silently absent from training |
| `data/decisions_packed` | `board_act` calls in the epoch |
| `data/supervised_frac` | share of tokens carrying loss; the rest is board text the model reads and never writes |
| `data/padding_frac` | packing waste |

## 3 · Training

### 3a · Per role

`RoleCrossEntropyLoss` fills `RoleMetricAccumulator` on device; the trainer drains and
all-reduces it in `_drain_role_metrics`. Every token is assigned one role by the encoder
(`board_harmony.BoardHarmonyEncoder.encode`); input and tool-observation tokens are ignored.

| metric | definition |
|---|---|
| `loss/decision`, `acc/decision` | the action JSON, i.e. the tool-call argument |
| `loss/format`, `acc/format` | the Harmony scaffolding the model must emit (` to=functions.board_act`, `<\|channel\|>commentary json`, `<\|call\|>`); format drift shows here before it shows in a sample |
| `loss/terminate`, `acc/terminate` | the closing hand-over, kept separate so it cannot dilute `decision` |
| `loss/reasoning`, `acc/reasoning` | the analysis channel; absent on the no-reasoning corpus |
| `tokens/<role>`, `tokens/<role>_frac` | supervision budget per role |
| `action/exact_match` | share of calls whose every decision token is the argmax |
| `action/count` | calls scored, over the whole logging interval |

### 3b · Per action

`ActionCapture` collects the argmax and gold ids of decision tokens on logged steps,
decodes both, and `board_metrics.compare` diffs them field by field. Counts are summed
across ranks before dividing (`metrics_from_counts`).

| metric | definition |
|---|---|
| `transition/exact` | the whole payload matches the gold, so the board that comes back is identical. The environment is a function of the action, so this is state-transition accuracy |
| `transition/tree` | verbs, molecules and every ranking's applied head match: the AND–OR graph evolves identically |
| `action/json_valid` | the decode parses as a legal `board_act` call |
| `action/verbs_match` | open / rank / done, in order |
| `action/n_items_match` | same number of actions in the call — the breadth decision |
| `action/mid_set_match` | same molecules, ignoring order |
| `action/mid_order_match` | same molecules in the same order |
| `action/mid_misorder` | set matches, order does not |
| `action/rank_head_match` | the applied candidate of each ranking |
| `action/rank_order_match` | the whole ranking, head and tail |
| `action/rank_misorder` | right head, wrong tail |
| `action/decoded` | calls decoded, on logged steps only, so it is a fraction of `action/count` by design |
| `action/capture_overflow` | decision tokens dropped by the capture's own cap |

Everything in 3a and 3b is teacher-forced: it says what the model would emit given the gold
prefix, not what it does when it drives the board itself. That is stage 4.

## 4 · Evaluation

### 4a · Per rollout

`eval_board_agent.py` writes one row per (target, rollout).

| field | definition |
|---|---|
| `solved` | the target closed: every leaf of some route is in the stock |
| `stop` | why the rollout ended: `final` (the model handed over), `terminate`, `budget`, `stuck` (nothing actionable and no route), `max_turns`, `illegal_action` / `unparsed_call` (rejection cap), `malformed_tool_call`, `context_exceeded`, `model_error` |
| `calls`, `turns` | single-step calls spent (only `open` is charged) and turns taken |
| `routes[]` | the routes the rollout claimed, each stamped with the budget spent when it was found |
| `errors{}` | the board's rejections, keyed by their message |
| `output[]` | the model's raw output per turn and the board's rejection text where there was one |
| `applied_steps[]`, `offered_steps[]` | the reactions applied, and every candidate the menus put on screen, with reaction names and bond changes |

### 4b · The table

`evaluation/eval_protocol/metrics/table_geo.py` scores every method's dump with the same code
at a fixed budget N_B of unique single-step calls; the protocol and the per-baseline budget
accounting are in `evaluation/eval_protocol/PROTOCOL.md`.

| column | definition |
|---|---|
| Success Rate | targets with at least one route whose leaves are all in the eMolecules stock, over all targets |
| Routes | distinct routes found within the budget; routes reaching the same leaf set count as one |
| Plausibility | share of the representative route's steps with p ≥ 0.05 |
| Round-Trip | share of the representative route's steps whose product the forward model recovers within top-5 |
| Price | the representative route's summed starting-material price, capped at P_max = $1000 and normalised as `log(1 + min(c, P_max)) / log(1 + P_max)` |
| Better-Than-Ref | how often the representative route beats the published reference route, over targets that have one |

The representative route of a target is, among its Pareto non-dominated routes on the three
objectives, the one with the highest mean of the three normalised scores. A target with no
route is not excluded: it is scored 0 on plausibility and round-trip and at the price cap.
