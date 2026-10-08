---
license: apache-2.0
base_model: openai/gpt-oss-20b
tags:
  - chemistry
  - retrosynthesis
  - agent
  - tool-use
  - gpt-oss
language:
  - en
pipeline_tag: text-generation
---

# RetroPlanner

RetroPlanner is a 20B gpt-oss model fine-tuned to **plan multi-step retrosyntheses by
operating a search board**. It does not propose disconnections itself: a single-step
retrosynthesis model supplies candidate disconnections, and RetroPlanner decides which
molecule to open, which candidate to take, when a route is finished, and when a branch is
dead. The thing it learned is the *control policy over a search*, not the chemistry of one
step.

That distinction matters for what you can do with it. Loading these weights and prompting
them like a chat model will not give you retrosynthetic routes. The model expects a board
rendered in a specific text format and a specific tool interface, both described below.

**Inference code:** <https://github.com/KU-AGI/RetroPlanner>. The board harness, the
single-step model servers and the runner that drive this checkpoint all live there; the
weights alone are only half of the system.

## At a glance

| | |
|---|---|
| Base | `openai/gpt-oss-20b` (MoE, 32 experts, 4 active per token) |
| Parameters | 20.9B total, ~3.6B active |
| Layers / hidden / heads | 24 / 2880 / 64 (8 KV) |
| Context | 131,072 |
| Precision | bf16 (**not** quantized — `quantization_config` is stripped) |
| Size on disk | 39 GB, 9 safetensors shards |
| Default single-step model | R-SMILES (`root_aligned`), top-10 |

## Serving

### The TRITON MoE backend is not optional

vLLM's FlashInfer CUTLASS path for an **unquantized** MoE produces fluent nonsense on these
weights **without raising an error**. A run served that way looks healthy while every
board decision it makes is garbage.

```bash
vllm serve KU-AGI/RetroPlanner \
  --served-model-name KU-AGI/RetroPlanner \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.70 \
  --enable-auto-tool-choice --tool-call-parser openai \
  --kernel-config '{"moe_backend":"TRITON"}'
```

Not every vLLM build accepts `--kernel-config`; if yours rejects it, use a build that does
rather than dropping the flag. **Always gate a run on a coherence probe** — ask the served
model something trivial and check the answer is sensible before spending GPU hours:

```bash
curl -s http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"KU-AGI/RetroPlanner","prompt":"The capital city of France is","max_tokens":6,"temperature":0}'
# expect "... Paris"
```

The GitHub repository pins `vllm==0.14.0` in its `requirements.txt`.

## Running the planner

Inference goes through the board harness in the GitHub repository. Its README covers
installation, and the steps below are the short version.

```bash
git clone https://github.com/KU-AGI/RetroPlanner && cd RetroPlanner
conda create -y -n retroplanner python=3.12 && conda activate retroplanner
pip install -r requirements.txt

# point the runner at these weights
huggingface-cli download KU-AGI/RetroPlanner --local-dir checkpoints/retroplanner
. config/env.sh            # RP_MODEL_DIR defaults to checkpoints/retroplanner

# single-step menu (R-SMILES) and forward model (ReactionT5v2, round-trip score)
cd tools/reaction-mcp
bash scripts/launch_ssr_fleet.sh
python scripts/menu_cache_proxy.py --port $RP_PORT_MENU \
    --cache data/route_search/draw_cache/rsmiles__d0__k10.json \
    --upstream http://127.0.0.1:$RP_PORT_RSMILES/predict &
bash scripts/feas_forward_fleet.sh
cd ../..

# serve the checkpoint on vLLM and plan routes for your targets
TARGETS_FILE=/path/to/targets.jsonl \
  bash evaluation/eval_protocol/runners/retroplanner_rsmiles.sh
```

`TARGETS_FILE` is JSON Lines, one target per line:

```json
{"id": 0, "target": "CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC", "gold_routes": []}
```

The runner stops every vLLM process on the machine before it serves the checkpoint. GPUs,
ports and conda env names come from [`config/env.sh`](https://github.com/KU-AGI/RetroPlanner/blob/main/config/env.sh).

### What the harness does around the model

- **Tool.** The model gets one function, `board_act`, whose argument is
  `{"actions": [ACTION, ...]}`. Each action is `open` (request disconnections for a
  molecule), `rank` (take one or more candidates at a molecule), or `done` (claim a
  finished route). The schema and the developer message are defined in
  [`evaluation/board/board/harmony.py`](https://github.com/KU-AGI/RetroPlanner/blob/main/evaluation/board/board/harmony.py).
- **Board.** After every call the harness returns the board as text: the open molecules,
  each one's candidate menu from the single-step model, and the scores shown next to each
  candidate (plausibility, round-trip from the forward model, and price).
- **Prompt format.** The harness does not go through the chat template. It renders the
  board history with `openai_harmony` (the exact training-time encoding, reasoning effort
  `medium`) and calls vLLM's raw `/v1/completions` endpoint with
  `skip_special_tokens: false`. Every turn, it appends `<|channel|>analysis<|message|>`
  to the prompt, so the model always writes its reasoning before acting. Without this,
  greedy decoding tends to skip the reasoning and go straight to the tool call.
- **Two calls per turn.** In training, the token after the analysis channel's `<|end|>`
  is not supervised. So the harness stops the first call at `<|end|>`, appends
  `<|start|>assistant` itself, and makes a second call for the tool call (`--two-stage`).
- **Restarts.** Each episode ends at the first complete route it finds. The harness then
  starts a new episode on a fresh board, sampling at temperature 0.6 with a different
  seed each time. This continues until the target has used up its budget of distinct
  molecules opened (`ITER_MAX_BUDGET`, default 500; a molecule already opened in an
  earlier episode is not counted again) or reached the rollout cap (`ITER_MAX_ROLLOUTS`,
  default 40). The routes from all episodes are pooled, so one target can return several
  alternative routes.

## Limitations

- The model only chooses among the disconnections the single-step model offers. If the
  right disconnection is missing from the menu, the model cannot find it.
- It was trained with R-SMILES menus and a specific board format. Other single-step
  models or board renderings are outside what it was trained on.
- Routes are proposals. Check them before using them in a lab.

## Citation

If you use RetroPlanner, please cite the repository:
<https://github.com/KU-AGI/RetroPlanner>.
