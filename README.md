# RetroPlanner

Multi-route expansion and multi-objective reasoning for LLM-based retrosynthesis
planning. This repository holds the evaluation harness, the single-step model servers
it runs on (R-SMILES by default), the data-construction pipeline, and the SFT module
that trains the RetroPlanner checkpoint (gpt-oss-20b).

## Installation

Two conda envs: one for evaluation (vLLM, R-SMILES, the board, the forward model and
the scorer share torch 2.9.1) and one for training (torchtitan needs a torch nightly).

```bash
# evaluation
conda create -y -n retroplanner python=3.12
conda activate retroplanner
pip install -r requirements.txt          # from the repository root

# training: see Training below for the torchtitan checkout
conda create -y -n torchtitan python=3.12
conda activate torchtitan
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

The scripts look the envs up under `$CONDA_ROOT` (the conda `envs/` directory). Set
it in [`config/env.sh`](config/env.sh) or export it. To use other env names, set
`RP_ENV` and `RP_ENV_TRAIN`.

The checkpoints, single-step weights, stocks and caches are not in git. They sit in
the repository at the paths the scripts expect, all of them listed in `.gitignore`.
[`config/import_worktree.sh`](config/import_worktree.sh) brings them in from an
existing tree.

## Evaluation

USPTO-190 with R-SMILES as the single-step model. [`PROTOCOL.md`](evaluation/eval_protocol/PROTOCOL.md)
gives the settings, along with what each baseline arm may do.

```bash
. config/env.sh

# 1. single-step menu and forward model
cd tools/reaction-mcp
bash scripts/launch_ssr_fleet.sh                      # R-SMILES replicas + proxy
python scripts/menu_cache_proxy.py --port $RP_PORT_MENU \
    --cache data/route_search/draw_cache/rsmiles__d0__k10.json \
    --upstream http://127.0.0.1:$RP_PORT_RSMILES/predict &
bash scripts/feas_forward_fleet.sh                    # ReactionT5v2, round-trip axis
cd ../..

# 2. the run: serves the checkpoint on vLLM and runs the board eval
bash evaluation/eval_protocol/runners/retroplanner_rsmiles.sh

# 3. the main table
bash evaluation/eval_protocol/runners/table_main.sh both
```

If the menu cache is missing, `menu_cache_proxy.py` builds it from the live R-SMILES
answers. Warning: `retroplanner_rsmiles.sh` stops every vLLM process on the machine
before it serves the checkpoint.

## Training

SFT runs as a torchtitan experiment. Our module and a patch to upstream live in
[`torchtitan/`](torchtitan/):

```bash
git clone https://github.com/pytorch/torchtitan && cd torchtitan
pip install -r requirements.txt                                  # in the torchtitan env
git apply "$RP_ROOT"/torchtitan/patches/ours.patch
rsync -a "$RP_ROOT"/torchtitan/torchtitan/ ./torchtitan/

NGPU=4 ./torchtitan/experiments/retroplanner/run_board_sft.sh    # CONFIG=sft_gpt_oss_20b_rs
```

Set `RP_ROOT` to this repository. [`torchtitan/RETROPLANNER.md`](torchtitan/RETROPLANNER.md)
covers the configs and the ablations.

## Repository

| directory | contents |
| --- | --- |
| [`evaluation/`](evaluation/) | the board harness, the runner and the scorer |
| [`tools/reaction-mcp/`](tools/reaction-mcp/) | the `reaction_mcp` package (bond changes, routes, templates, stock, the scores the board shows (`scoring/`: plausibility, round-trip, price), the single-step model pool and the MCP server), the single-step servers and their launchers, and the data-construction pipeline in `scripts/` |
| [`tools/common/`](tools/common/) | error and formatting helpers the MCP server imports |
| [`torchtitan/`](torchtitan/) | the SFT module |
| [`config/`](config/) | machine settings: conda envs, GPUs, ports |
