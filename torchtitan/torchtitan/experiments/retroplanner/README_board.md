# Board-format SFT — gpt-oss-20b on tool calls against the AND-OR board

The format the planner is trained on. Unlike the `<think>/<act>` trajectories of
`README.md`, an episode is a Harmony **tool-call** conversation: the board arrives on the
`commentary` channel from `functions.board_act`, and every assistant turn is a
call whose argument is JSON.

```
user                     the opening board
assistant -> board_act   {"actions": [{"type": "rank", "mid": "vw", "order": [1, 0, 4]}]}
functions.board_act      the board that results
...
assistant                the route it hands over                     (final channel)
```

Everything except the dataloader is shared with the trajectory format: the
trainer, the role-split loss, the accumulator, the parallelism, the checkpoint
preflight.

## Running it

```bash
# 1. build the corpus (rdkit env)
python tools/reaction-mcp/scripts/render_board_episode.py --all \
    --routes-per-target 8 --rank-max 3 --menu-show 10 \
    --signals q,p,rt --axis p --cutoff 0.05 \
    --format harmony --deciding p --analysis none --raw-text template \
    --date 2024-06-01 --strict --no-evidence \
    --out .../sft_board/board.harmony.jsonl

# 2. put the WIRE preamble on each row (needs openai_harmony)
"$CONDA_ROOT"/vllm-new/bin/python \
    tools/reaction-mcp/scripts/verify_harmony.py \
    .../sft_board/board.harmony.jsonl \
    --augment .../sft_board/board.wire.jsonl

# 3. validate and split BY TARGET
python -m torchtitan.experiments.retroplanner.scripts.prepare_board_dataset \
    --input .../sft_board/board.wire.jsonl \
    --out-prefix .../sft_board/board \
    --assets "$GPT_OSS_ASSETS" --seq-len 16384 --val-frac 0.02

# 4. smoke, then train
CONFIG=sft_gpt_oss_20b_rs_smoke ./torchtitan/experiments/retroplanner/run_board_sft.sh
./torchtitan/experiments/retroplanner/run_board_sft.sh
```

Step 2 is not optional and not cosmetic. vLLM builds the developer message with
`openai_harmony`; the HF chat template builds it from the same tool schema and
its TypeScript namespace differs. Training on
the template's preamble and serving with vLLM means the model never sees the tool
declaration it was trained on. `BoardHarmonyEncoder` therefore reads
`wire_system` / `wire_developer`, and a test asserts the encoded stream equals
the openai_harmony rendering token for token — except the last one, where
training ends with `<|return|>` (the terminator the model must learn) and
`render_conversation` ends with `<|end|>` (a context terminator).

## What gets logged

### Per-role, from the loss (device-side, no sync)

| metric | meaning |
|---|---|
| `loss/decision`, `acc/decision` | the action JSON — the tool-call argument |
| `loss/format`, `acc/format` | the harmony scaffolding: recipient, channel, `<\|call\|>`. Format drift shows here first |
| `loss/terminate`, `acc/terminate` | the closing route report. Its own role so it cannot dilute `decision` |
| `loss/reasoning` | the analysis channel, when the corpus carries one. Absent for the non-reasoning set |
| `tokens/<role>`, `tokens/<role>_frac` | supervision budget. A small fraction of an episode is supervised; the rest is board the model reads and never writes |
| `action/exact_match` | every token of a call is the argmax |

### Per-action, from the capture (decoded on logged steps only)

`action/exact_match` says a call was wrong. These say **how**, which is the
difference between a chemistry error and a formatting one:

| metric | meaning |
|---|---|
| `transition/exact` | the whole payload matches, so the board that comes back is byte-identical. The environment is a function of the action, so this **is** state-transition accuracy |
| `transition/tree` | verbs, molecules and every ranking's HEAD match. The AND-OR tree evolves identically; only the recorded intent differs |
| `action/json_valid` | the decode parses as a legal `board_act` call at all |
| `action/verbs_match` | open / rank / dead / done, in order |
| `action/n_items_match` | the call carries the same number of actions — the breadth decision |
| `action/mid_set_match` | the same molecules, ignoring order |
| `action/mid_order_match` | the same molecules in the same order |
| `action/mid_misorder` | **set matches, order does not** — a scheduling slip, not a wrong molecule |
| `action/rank_head_match` | the applied candidate. The only part of a ranking that moves the tree |
| `action/rank_order_match` | the whole ranking, head and tail |
| `action/rank_misorder` | right head, wrong tail: the same move, a different declared fallback |
| `action/mid_hallucinated` | a named molecule is not in the board's OPEN block (validator/prepare only — the board is not in the training tensors) |
| `action/candidate_off_screen` | a candidate number past the end of the menu shown |
| `action/decoded` | how many calls the rates are over. A rate over three calls is not a measurement |

All of it is teacher-forced: "would the model emit this token given the gold
prefix", the same footing as `acc/*`.

**Two different denominators, on purpose.** `action/count` and
`action/exact_match` come from the device-side accumulator and cover the whole
logging interval. `action/decoded` and everything under `transition/*` come from
the capture, which is armed only when `metrics_processor.should_log(step)` is true
-- it copies the decision tokens to host, and arming it every step would put a
sync in the hot path. So at `log_freq=5` `action/count` covers five steps and
`action/decoded` the one that was logged; they are not meant to match. What IS a
cross-check is `action/exact_match` against `transition/exact`, because they
measure nearly the same thing by different routes, one on device over the
interval and one by decoding the sample.

Both are summed across ranks before dividing, so a rank-local count is never read
against a cross-rank one.

**Held out.** `RetroValidator` runs the same role and capture metrics over the
val split (`val/*`, split by target) and over a depth-stratified train subset
(`trainprobe/*`, from `scripts/make_probe_subsets.py`). The two are stratified to
the same depth mix, so a gap between them reads as memorisation rather than as a
difference in how deep their episodes are.

Two failure modes the capture handles rather than hides: activation checkpointing
recomputes a chunk in backward, so the same `(span, position)` can arrive twice
and is keyed rather than appended; and the gold text is decoded from the LABEL
stream, so a span-extraction bug shows up as `action/json_valid` collapsing on
the gold side rather than as a plausible-looking model failure.

### Before training — `scripts/prepare_board_dataset.py`

Every check is for something silent at training time. A call naming a molecule
the board never showed produces the same loss as a call naming the wrong one.

| group | what it asks |
|---|---|
| `check/action/unparsed` | does the call parse as a legal `board_act` |
| `check/state/mid_not_open` | does it name a molecule listed under OPEN |
| `check/state/candidate_off_screen` | does a ranking name a candidate past the menu |
| `check/action/two_actions_one_molecule` | two actions on one molecule in one call |
| `check/final/no_route` | the final message hands over no route |
| `length/dropped_at_seq_len` | episodes the packer would drop whole |
| `data/routes_per_target_mean` | how many of a target's diverse routes made it in |

The split is **by target**, never by instance: a target's several routes share
every board they walk, so splitting by instance puts the same observation on both
sides.

## Corpus

Each training target is searched once with Retro*-0; the searched routes are
Pareto-ranked over plausibility, round-trip and starting-material price, and up
to eight routes with distinct starting-material sets from the first two fronts
are merged at their shared intermediates into one DAG. Replaying that DAG
breadth-first on the board turns every step into an `open`, a `rank` (with how
many candidates to commit) or a `done`, and the episode ends with the final
handover. `row["route"]` / `row["route_queue"]` keep the axes each route was
selected on (`plaus_min`, `rt1`, `cost_usd`, `lls`, `pareto_rank`).

The board's depth bound has to cover a route's longest linear chain (`lls`), or
the board cannot represent the route at all.

A turn with `supervised: false` is a refused action -- a wrong call inserted in
front of a label together with the board's refusal, so the corpus contains
recovery. It is encoded as `ROLE_IGNORE`: in the sequence for coherence, out of
the loss.
