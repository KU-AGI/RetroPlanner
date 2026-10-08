# reaction-mcp environments

**Default: one env.** Everything RetroPlanner's evaluation runs here (the R-SMILES SSR
fleet, the board drivers, the forward fleet, the proxies, the MCP server) shares the
`retroplanner` env built from the repository's `requirements.txt` (torch 2.9.1, as
vLLM pins it); `config/env.sh` points every `RP_ENV_*` role at it. The table below is
the older per-model layout. It is still needed for `rsmiles_server.py` (`rsmiles` env)
and the opt-in LocalRetro servers (`localretro` env).

The two single-step models (R-SMILES and LocalRetro) and the MCP server each pin
their own torch, so they run in separate conda envs under `$CONDA_ROOT`
(`config/env.sh`). Scripts resolve an
interpreter with `rp_py <env>`, which fails at launch if the env is missing.

These are the only envs the code under `tools/reaction-mcp/` uses.

| env | python | torch | rdkit | used by |
|-----|--------|-------|-------|---------|
| `sci-tools-mcp` | 3.11 | 2.13 (cu130) | 2023.09 | the MCP server (`python -m reaction_mcp`), `singlestep_pool_server.py`, the stdlib proxies (`pool_proxy.py`, `menu_cache_proxy.py`, `ssr_shim.py`), `run_reasoning.sh` |
| `rsmiles` | 3.10 | 2.0.1 (cu118) | 2023.09 | `rsmiles_server.py` (OpenNMT-py 2.2, pool contract, port 8100); `launch_replicas.sh rsmiles` |
| `localretro` | 3.11 | 2.4.1 (cu121) | 2026.03 | `localretro_server.py` (DGL 2.4 + dgllife, pool contract, port 8097); `launch_replicas.sh localretro` |
| `syntheseus-gpu` | 3.11 | 2.8.0 (cu128) | 2026.03 | the SSR fleets: `launch_ssr_fleet.sh` (`predict_standalone.py root_aligned`) and `launch_ssr_fleet_localretro.sh` (`predict_standalone.py localretro`) |
| `verl-vllm` | 3.12 | 2.9.1 (cu128) | 2026.03 | route search and scoring (`traj_route_search.py`, `traj_route_label.py`, `traj_route_score.py`, `feas_cost.py`), the ReactionT5v2 forward fleet (`feas_forward_fleet.sh`), `run_reasoning_local.sh` |

Notes:

- **The SSR env's torch must match the CUDA driver.** A torch built for a newer
  CUDA than the driver supports cannot initialise CUDA, so a fleet launched in it
  runs on CPU while reporting itself healthy. Override with `SSR_PY=` only for an
  env whose torch matches the driver.
- **`rsmiles_server.py` vs the R-SMILES SSR fleet.** Both serve R-SMILES, but with
  different code: `rsmiles_server.py` runs OpenNMT directly in the `rsmiles` env
  and speaks the pool contract (`{task, product_smiles, top_k}`); the SSR fleet
  runs syntheseus' `RootAlignedModel` in `syntheseus-gpu` and speaks
  `{smiles, top_n}`. Route search and the board use the SSR fleet.
- `rsmiles_server.py` reads `REACTION_RSMILES_ENV` (default `rsmiles`) for the
  env whose `onmt_translate` it calls.
