# reaction-mcp

Retrosynthesis + reaction-analysis tools: single-step retro prediction over two
models (R-SMILES and LocalRetro, cross-model voting), atom mapping, bond-change
and reaction-center extraction, functional-group detection, and named-reaction
template lookup.

## Tools (`reaction_*`) — 11 active

### Prediction pool (`pool/`)

| Tool | Status | Backend |
|---|---|---|
| `reaction_predict_singlestep_retro` | real (pool) | one-shot single-step retro: fan-out to the live retro backends (`rsmiles`, `localretro`) + cross-model voting (selection backbone; `union` = pass@k) |
| `reaction_route_target` | real (pool + RDKit) | **single-vs-multi advisor**: does a single-step disconnection reach stock? → `in_stock` / `single_step` / `single_then_escalate` / `multi_step_search`, with a 0-100 complexity score as a prior. Advisory, not a gate — the LLM decides when to expand |
| `reaction_atom_map_reaction` | real | `rxnmapper` (IBM transformer) |

There is no multi-step, forward or condition tool. Multi-step route search runs
outside the MCP server, over the SSR fleets (`scripts/traj_route_search.py`).

### CGR topology (`cgr.py`, powered by CGRtools)

Read the Condensed Graph of Reaction (auto atom-mapped via RXNMapper when the
input is unmapped) to expose reaction-center structure to the model.

| Tool | Status | What it returns |
|---|---|---|
| `reaction_get_bond_changes` | real (CGRtools) | net bonds formed/broken/order-changed + inferred reaction type + FG labels. Single mode (`reaction_smiles="R>>P"`) or batch mode (`product_smiles` + `candidates` precursor sets) |
| `reaction_extract_reaction_center` | real (CGRtools) | reaction center atoms/bonds with per-atom **rehybridization / neighbor / charge** changes + radius-0 template, plus a graded `local_environment` (reaction center + 1..`max_radius` bond-shells; default 2, cap 3) |

### Functional groups + named-reaction templates

| Tool | Status | What it returns |
|---|---|---|
| `reaction_detect_functional_groups` | real (RDKit) | prioritized named functional groups with reactivity labels (`fg_type`, `atom_indices`, `reactivity_note`) — which bonds are likely disconnection sites |
| `reaction_match_templates` | real (RDKit + named corpus) | matches a molecule against a corpus of **named reactions** (Rxn-INSIGHT SMIRKS, vendored at `reaction_mcp/data/named_reaction_smirks.json`, ~470 distinct names) and returns the named reactions that could produce it (e.g. *Suzuki coupling*, *Buchwald-Hartwig N-arylation*, *Esterification of Carboxylic Acids*), ranked by match specificity. The first layer so the model never authors SMARTS — it picks from real, human-named reactions. `name_filter` narrows to a specific reaction; each result's `template` SMIRKS can be fed into `reaction_check_template_compatibility`. Override the corpus with `REACTION_TEMPLATE_LIBRARY` (a `*.json` named corpus, or a tab-separated `*.csv.gz` retro-template file for unnamed breadth). |
| `reaction_check_template_compatibility` | real (RDKit) | verifies whether a molecule matches a specific retro/forward SMARTS template — `{compatible, matched_atoms, match_count, template_class}` |
| `reaction_extract_reaction_template` | real (rdchiral) | extracts a *fresh* generalized retro template (`product>>reactants` SMARTS) from a given reaction `R>>P` — generalizes beyond the named corpus and feeds straight into `reaction_check_template_compatibility` |

### Disconnection proposal (`disconnection.py`, BRICS)

| Tool | Status | What it returns |
|---|---|---|
| `reaction_propose_disconnections` | real (RDKit BRICS) | candidate retro disconnection **bonds** on a target: each with `bond`, a rule-based `bond_type` (amide/ester/C–N/ether/biaryl…) from the BRICS L-type pair, the two `synthons` from cutting that bond, and their `synthon_sizes` (heavy-atom counts). **Hint generator, not a decider** — BRICS is context-blind and never cuts fused rings, so it *proposes* leads that must be confirmed with `reaction_predict_singlestep_retro` / `reaction_match_templates`. Absence of a cut ≠ hard to make (ring-forming disconnections are invisible here). Scoped deliberately narrow: no building-block/stock matching. |

### Stock

| Tool | Status | What it returns |
|---|---|---|
| `reaction_in_stock` | real | purchasable-building-block lookup against the configured stock (`REACTION_STOCK` or the `stock=` arg) |

## Model pool (`reaction_mcp/pool/`)

Single-step retro is served by two models. Each is wrapped as a `Predictor`
advertising which `Task`s it serves; the `ModelPool` fans a request out to every
live backend, and the `consensus` layer aggregates their candidates into one
ranked list by **cross-model voting** — candidates proposed by both models rank
higher (ties broken by within-backend rank). A mass guard drops precursor sets
that cannot supply the product's skeleton before the vote.

Backends run as **warm HTTP microservices in their own conda envs**
(`docs/ENVIRONMENTS.md`), discovered via env vars.

| Backend | Task | Transport | Model / env |
|---|---|---|---|
| `rsmiles` | retro | HTTP `REACTION_RSMILES_URL` (:8100) | R-SMILES (OpenNMT-py 2.2), `scripts/rsmiles_server.py`, `rsmiles` env |
| `localretro` | retro | HTTP `REACTION_LOCALRETRO_URL` (:8097) | LocalRetro (OpenRetro), `scripts/localretro_server.py`, `localretro` env |

Each backend is gated by `REACTION_ENABLE_<NAME>` (both on by default). To bring
them up:

```bash
python scripts/rsmiles_server.py --port 8100        # rsmiles env
python scripts/localretro_server.py --port 8097     # localretro env
# or N replicas behind a least-outstanding proxy, URL recorded in scripts/replica_urls.env:
bash scripts/launch_replicas.sh rsmiles 8
bash scripts/launch_replicas.sh localretro 8
```

The pool's default URLs already point at the single-server ports, so once a
server answers `/health` the pool uses it automatically. `scripts/singlestep_pool_server.py`
exposes the same consensus on the SSR wire contract (`{smiles, top_n}`) for the
route-search drivers.

**Forward round-trip.** `round_trip_filter` (in `consensus.py`) checks whether a
forward model regenerates the product from each precursor set. No forward
backend is configured, so with `round_trip=True` / `REACTION_RETRO_VERIFY=soft|hard`
every candidate is marked `round_trip.matched = null` and the order is left
alone. Adding a `Task.FORWARD` `BackendSpec` in `config.py` re-enables it.

### Retro candidate selection (env, read at call time)

Set before launching the reaction server to fix how every retro call combines
per-model candidate lists (see `pool/consensus.py`):

- `REACTION_RETRO_SELECTION` — `consensus` (default) | `single` | `plurality` | `rrf` | `union`
- `REACTION_RETRO_SINGLE_MODEL` — for `single`: which model (default `rsmiles`; fail-closed, `auto` = best available)
- `REACTION_RETRO_VERIFY` — `off` | `soft` | `hard` (forward round-trip; a no-op without a forward backend)
- `REACTION_RETRO_GATE_TAU` — int; abstain when fewer than TAU models agree on top-1 (default 0 = off)
- `REACTION_RETRO_DEPTH` — per-model fan-out depth (default = final `top_k`)
- `REACTION_RETRO_TOPK` — final number of candidates returned (overrides the tool's `top_k`)

### Complexity routing

`reaction_route_target` decides at a node whether one step is enough. Its
criterion is **operational** (`probe=True`, default): does a single-step
disconnection reach purchasable stock? It stock-checks the target, runs
single-step retro, and checks stock membership of the precursors →

- `in_stock` — target is already buyable (a leaf; don't expand);
- `single_step` — a one-step disconnection lands *entirely* in stock;
- `single_then_escalate` — reaches stock *partially*; take the step, recurse on
  the non-stock precursors;
- `multi_step_search` — no precursor is buyable → target is far from stock and
  needs a route search.

For `single_step` / `single_then_escalate` the result carries a
**`chosen_disconnection`** — the precursor set to take, split into `leaves`
(buyable, done) and `recurse_on` (the next nodes) — so the caller acts without
re-running single-step retro.

The molecular-complexity score (`complexity.py`: SAscore·ring·fsp3·macrocycle·
stereocenters, weighted-sum → 0-100) is returned as a **prior only**. Complexity
does **not** by itself predict route depth — route need is a
*target-to-stock-gap* question, not an absolute-complexity one. So the score is
advisory / a cheap fallback (`probe=False`, or when no stock / single-step
backend is live); the operational probe is authoritative.

The node loop itself (pop node → `route_node` → act on the decision → recurse on
`chosen_disconnection.recurse_on`; visited-set + step budget) is encoded as a
reference driver in `reaction_mcp/trajectory.py` (shared core in
`reaction_mcp/routing.py`, used by both the driver and the `reaction_route_target`
tool). `multi_step_search` nodes are reported as unresolved:

```bash
REACTION_STOCK=paroutes_n1 python -m reaction_mcp.trajectory "CC(=O)Nc1ccc(-c2ccccc2)cc1"
```

### Performance

- **Startup warmup.** The server loads rxnmapper and compiles the named-reaction
  template corpus (~470 SMIRKS) in a background thread at boot, so the first
  atom-map call is not a cold start. Disable with `REACTION_WARMUP=0`.
- The retro servers hold a global lock around `predict`, so one server answers
  one request at a time; `launch_replicas.sh` puts N replicas behind
  `pool_proxy.py` for concurrent callers.

All warm servers share the contract in `scripts/pool_server.py`
(POST task request → `{"candidates": [...]}`, `GET /health`).

## Models & references

| Role | Model | Reference |
|---|---|---|
| Atom mapping | **RXNMapper** | Schwaller et al., *Sci. Adv.* 2021 — [10.1126/sciadv.abe4166](https://doi.org/10.1126/sciadv.abe4166) |
| Retro (1-step) | **R-SMILES** (root-aligned SMILES) | Zhong et al., *Chem. Sci.* 2022 — [10.1039/D2SC02763A](https://doi.org/10.1039/D2SC02763A) |
| Retro (1-step) | **LocalRetro** | Chen & Jung, *JACS Au* 2021 — [10.1021/jacsau.1c00246](https://doi.org/10.1021/jacsau.1c00246) |
| Named-reaction template match | **Rxn-INSIGHT SMIRKS corpus** | Dobbelaere et al., *J. Cheminform.* 2024 — [10.1186/s13321-024-00834-z](https://doi.org/10.1186/s13321-024-00834-z) |

## Conda env

The MCP server runs in the `sci-tools-mcp` env (Python 3.11; fastmcp, rdkit,
rxnmapper, CGRtools, rdchiral). `tools/common` is imported from `tools/`, so run
it from `tools/reaction-mcp`:

```bash
cd tools/reaction-mcp
$CONDA_ROOT/sci-tools-mcp/bin/python -m reaction_mcp
```

See `docs/ENVIRONMENTS.md` for the model envs.

## Example

```jsonc
// single-step retrosynthesis for aspirin
{"product_smiles": "CC(=O)Oc1ccccc1C(=O)O", "top_k": 5}
```
