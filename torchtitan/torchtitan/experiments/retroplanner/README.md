# retroplanner — role-aware SFT of gpt-oss on retrosynthesis agent trajectories

Supervised fine-tuning of `gpt-oss-20b` on multi-turn AND-OR-tree retrosynthesis
planning episodes, with the loss reported separately for the two jobs the model
has to do: **reasoning** (the chain of thought) and **decision** (the action).

## Why this is not `ChatDataLoader`

The stock chat dataset handles a single `[user, assistant]` pair. These records
are whole episodes, every assistant turn a supervision target, so this experiment does its own harmony rendering and carries two extra
per-token streams through packing and into the loss.

## What gets logged

### Before training — `scripts/prepare_dataset.py`

Every record is checked and the counts are written to a report (and optionally
to a wandb run of `job_type=dataset`, so the corpus a run consumed is recorded
next to the run):

| group | what it asks |
|---|---|
| `format/*` | does every assistant turn have the `<think>…</think>\n<act>…</act>` shape, and does the action parse as `single ID` / `rank ID c<a>,c<b>,c<c>` / `done`? |
| `action/*` | does the parsed action agree with the record's own `actions` list and `n_expansions`? |
| `gold/*` | do the fragments of the committed candidate match the children the search tree recorded? Split into an outright contradiction, a tree that repeats a reagent, and a tree that carries an extra piece. |
| `state/*` | is each user message the correct successor of the previous action — turn index, expansion budget, the echo of an expansion, the pieces a commit opens, `done` only when nothing is open? |
| `length` | how many trajectories the packer would drop at a given `seq_len`. |

Compare fragments with `--canonicalize inchikey14` (needs RDKit). A plain string
comparison reports tautomers — a 2-pyridone written as a 2-hydroxypyridine — as
gold-adoption failures.

### During training — per logged step, to wandb

| metric | meaning |
|---|---|
| `loss/reasoning`, `loss/decision`, `loss/format` | cross-entropy per token, split by role |
| `acc/reasoning`, `acc/decision`, `acc/format` | teacher-forced top-1 accuracy per role |
| `action/exact_match` | fraction of actions whose **every** token is the argmax — gold adoption under teacher forcing. This is the number that says whether the model would have picked the same disconnection. |
| `action/count` | actions scored in the interval |
| `tokens/<role>`, `tokens/<role>_frac` | supervision budget per role |
| `data/*` (per epoch, in the log) | trajectories packed vs dropped, supervised-token fraction, padding fraction |

`loss/format` is the tell for format drift: it covers the harmony channel
headers and terminators, so if the model starts losing the
`<|channel|>final<|message|>` structure it shows up there before it shows up in
a sample.

## Sequence layout

One trajectory becomes one packed sequence. Each turn renders as the two
messages the model actually emits at inference:

```
<|start|>user<|message|>STATE<|end|>
<|start|>assistant                                   <- prompt, not supervised
<|channel|>analysis<|message|>THINK<|end|>           <- format | reasoning | format
<|start|>assistant<|channel|>final<|message|>ACT<|end|>   <- format | decision | format
```

with `<|return|>` closing the last turn. The `<think>`/`<act>` tags of the
source data are dropped: harmony separates reasoning from the answer by
channel, so the channel *is* the tag.

Because the whole episode is one causal sequence, earlier turns' analysis stays
in context. The gpt-oss chat template drops prior chain-of-thought when
rendering for inference, so **the serving loop must keep it** for training and
inference to agree. If you serve with a harness that strips prior analysis,
switch to per-turn expansion instead, at a multiple of the tokens.

`positions` restarts at 0 on each trajectory; the varlen/flex attention
backends use that to stop packed trajectories from attending to each other.

## Running it

```bash
# 1. validate and split the corpus
python -m torchtitan.experiments.retroplanner.scripts.prepare_dataset \
    --input "$RP_ROOT"/tools/reaction-mcp/data/train_clean/\
sft_distilled/conv_full7_k50_paroutes.oss.jsonl \
    --canonicalize inchikey14 \
    --seq-len 16384 \
    --hf-assets-path "$GPT_OSS_ASSETS" \
    --wandb

# 2. smoke the pipeline (10 steps at full seq_len, no wandb, no checkpoint)
CONFIG=sft_gpt_oss_20b_rs_smoke ./torchtitan/experiments/retroplanner/run_sft.sh

# 3. train
./torchtitan/experiments/retroplanner/run_sft.sh
```

`run_sft.sh` sets the wandb environment (`WANDB_PROJECT`, `WANDB_RUN_NAME`,
`WANDB_RUN_GROUP`, …) that torchtitan's `WandBLogger` reads. Set
`WANDB_MODE=offline` to record locally.

## Environment notes

On a host whose driver cannot run the CUDA-13 wheels torchtitan main targets,
the `torchtitan` conda env uses a CUDA-12 torch nightly. Three gaps are closed
from inside this folder -- core is untouched:

- **`ChunkedLossWrapper` breaks on that torch.** Its per-chunk backward
  splices gradients back through a custom autograd Function, and that dies with
  `RuntimeError: The tensor has a non-zero number of elements, but its data is
  not allocated yet`. This is not specific to this experiment: stock
  `llama3_debugmodel_varlen_attn` fails identically inside core's own
  `trainer.py`, while the otherwise-identical `llama3_debugmodel_ce_loss`
  (unchunked) trains fine. `CheckpointedChunkedLoss` in `loss.py` replaces it,
  keeping the same one-chunk-of-logits memory profile via stock
  `torch.utils.checkpoint`.
- **`create_block_mask(separate_full_blocks=...)`** does not exist in that
  torch; `compat.py` drops the argument. It only picks a kernel iteration order.
- **`torch.distributed.set_timeout`** does not exist either; `compat.py`
  aliases it to `_set_pg_timeout`. Without it the run dies *after* the last
  step, having done all the work.

The `varlen` backend needs FA3 (`flash_attn_interface`) built for the GPU in
the `torchtitan` conda env (`$CONDA_ROOT/torchtitan`); there is no build script
checked in here. `flex` is the default; override with `RETRO_ATTN_BACKEND`.

## Constraints

- **No tensor parallelism.** Per-role attribution needs replicated logits;
  vocab-parallel CE cannot say which role a token belonged to. The trainer
  raises rather than reporting a wrong split. Set `log_role_metrics=False` to
  train with TP and give up the split.
- **No pipeline parallelism**, for the same reason: PP drives the loss through
  its own schedule.
- FSDP2 is the intended configuration, and is what the board configs use.

## Layout

| file | role |
|---|---|
| `trajectory.py` | parsing + the format / gold / state checks |
| `harmony.py` | trajectory to harmony tokens, labels, role ids, decision span ids |
| `dataset.py` | greedy packing, `Stateful` dataloader |
| `loss.py` | sum-reduced CE plus the role/action accumulator |
| `trainer.py` | routes the role streams to the loss and the metrics to wandb |
| `config_registry.py` | the board SFT configs (see `README_board.md`) |
| `scripts/prepare_dataset.py` | validate, split, report |
| `compat.py` | shims for torchtitan main against an older torch |
| `tests/test_retroplanner.py` | CPU tests for alignment, masking and the accumulator |
