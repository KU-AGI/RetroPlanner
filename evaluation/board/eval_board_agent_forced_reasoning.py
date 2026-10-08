#!/usr/bin/env python
"""Closed-loop USPTO-190 eval, with the analysis channel FORCED open every turn.

Under greedy decoding (temperature 0) a schema-reasoning checkpoint served through
`eval_board_agent.py` never opens the analysis channel -- but not because the model
cannot reason. The analysis channel carries real probability at the first token of
a turn, just never the most probable one: the model DOES know how to write the
schema-form reasoning, it just loses the "open the channel or not" choice to the
tool-call branch every time under argmax. This script does not touch that coin flip -- it removes it, by injecting
the channel-open tokens into the prompt directly and letting the model complete
the reasoning and the following tool call greedily from there.

This is a SEPARATE script, not a flag on `eval_board_agent.py`, because the
mechanism is fundamentally different: that script calls the OpenAI-compatible
CHAT endpoint and lets the server decide the channel; this one renders the board
history through openai_harmony (the exact training-time encoding, not a
regenerated chat template) and calls the RAW completions endpoint with the
forced prefix `<|channel|>analysis<|message|>` appended to the prompt, then
parses the reasoning text and the tool call that follows out of one continued
greedy decode. `skip_special_tokens: false` is required on the request or the
channel/end/call markers this parsing depends on come back as empty strings,
which reads as a model that never reasons.

Board mechanics (LiveWorld, Board, action parsing) are IMPORTED from
`eval_board_agent.py`, not copied -- one definition of the environment, whichever
script drives it.

Usage:
  python "$RP_BOARD"/eval_board_agent_forced_reasoning.py \\
      --targets "$RP_PROTOCOL"/targets_uspto190.jsonl \\
      --model-url http://127.0.0.1:$RP_PORT_LLM/v1 --model retroplanner \\
      --menu-url http://127.0.0.1:$RP_PORT_MENU/predict --stock emols \\
      --developer-file "$RP_PROTOCOL"/developer/dev_retroplanner.txt \\
      --rt live:http://127.0.0.1:$RP_PORT_FORWARD --two-stage \\
      --illegal-cap 1 --workers $RP_WORKERS --out results/forced_reasoning.jsonl
"""
from __future__ import annotations

import os
import argparse
import importlib.util
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

SD = Path(__file__).resolve().parent
# config/paths.py, loaded by file location under its own name: the analysis pipeline has
# a module called `paths` of its own, and that one must keep the name.
_spec = importlib.util.spec_from_file_location("rp_paths", SD.parents[1] / "config" / "paths.py")
RP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RP)
# traj_route_common / verify_harmony live in tools/reaction-mcp/scripts; it goes behind SD.
sys.path.insert(0, str(Path(RP.MULTISTEP) / "scripts"))
sys.path.insert(0, str(SD))
# reaction_mcp (the axis scorers, reaction_mcp.scoring) is importable from tools/reaction-mcp.
sys.path.insert(0, str(Path(RP.MULTISTEP)))

from board import harmony as H              # noqa: E402
from board import render as R                # noqa: E402
from board.episode import load_node_scores   # noqa: E402
from board.parse import ActError, Open, parse_act_json  # noqa: E402
from board.state import Board, BoardError, NoMenu, Terminate  # noqa: E402

import traj_route_common as C                 # noqa: E402
import verify_harmony as V                    # noqa: E402
from eval_board_agent import LiveWorld        # noqa: E402  (reused, not copied)

from openai_harmony import HarmonyEncodingName, Role, load_harmony_encoding  # noqa: E402

ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
FORCED_PREFIX = "<|channel|>analysis<|message|>"
ANALYSIS_END = "<|end|>"
ASSISTANT_TURN = "<|start|>assistant"
CALL_MARK = "<|channel|>commentary"
FINAL_MARK = "<|channel|>final"
MSG_MARK = "<|message|>"
_EFFORT = ["medium"]
_ILLEGAL_CAP = [1]
# Whether to treat the counter as CONSECUTIVE or CUMULATIVE.  Default is consecutive.
#
# RetroAgent counts `turns_since_last_expand`, i.e. only consecutive failures since the
# last success -- slime_retro/eval_worker.py resets it to 0 on every successful expand.
# A cumulative counter makes the same cap much stricter over a long episode.  The streak
# resets to 0 on every successful action, which makes the definition match RetroAgent.
# BOARD_ILLEGAL_CONSECUTIVE=0 selects the cumulative rule.
_ILLEGAL_STREAK = [os.environ.get("BOARD_ILLEGAL_CONSECUTIVE", "1") == "1"]
_SEED_BASE = [1234]
_STOP_EP_ON_SOLVE = [False]
"""End the episode the moment the board solves, instead of running it to the cap.

Off by default. It matters for the restart protocol: the episode does not reliably
stop at a solve on its own, so a solved episode can keep spending calls after the
answer was already found until a cap ends it. That tail buys nothing the next restart
would not buy more cheaply, so turning this on converts it into more restarts under
the same budget."""
_MAX_OPENS = [0]
"""Opens allowed per turn; 0 = unlimited, the trained behaviour.

The multi-open slot exists so one turn can commit a whole batch of molecules, and the
claim it is meant to support is that batching finds routes in fewer calls. That claim
cannot be read off a run that allows it, because target difficulty drives both the
opens and the batching. Setting this to 1 serialises the same intent -- extra opens in a
turn are dropped, and the agent sees the board again and may reopen -- so the two arms
differ in batching alone and the cost difference is attributable."""

_SKIP_OPEN_REASONING = [False]


class RawClient:
    """One /v1/completions call per turn, cycling replicas like ChatClient does."""

    def __init__(self, bases: list[str], model: str, max_tokens: int, timeout: float = 600.0,
                 temperature: float = 0.0, two_stage: bool = False,
                 ctx_len: int = 131072):
        from itertools import cycle
        self._bases = cycle(bases)
        self.two_stage = two_stage
        self.ctx_len = ctx_len
        self._lock = threading.Lock()
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.temperature = temperature
        self.stats: Counter = Counter()

    def complete(self, prompt: str, *, temperature: float | None = None,
                 seed: int | None = None, stop: list[str] | None = None) -> dict:
        """One raw completion.

        `temperature` defaults to greedy, which is the RetroAgent/Retro-R1 comparison
        setting and what a solve rate should be measured at. It is an argument because
        the iterative protocol restarts an episode on a fresh board with the same
        prompt: at temperature 0 that redraws the same trajectory, so restarts cost
        budget and return nothing new. Sampling is what makes a restart a second
        opinion rather than a replay, and `seed` varies per rollout so the restarts
        differ from each other and the run still reproduces."""
        with self._lock:
            base = next(self._bases)
        # vllm rejects prompt + max_tokens > max_model_len with a 400, and the driver
        # books a 400 as `model_error` -- an episode that reads as a model failure when
        # it is the harness overdrawing the window. It hits the LONG episodes, so it lands
        # hardest on exactly the targets a good checkpoint keeps alive. Ask for what is left.
        n_prompt = len(ENC.encode(prompt, allowed_special="all"))
        room = self.ctx_len - n_prompt - 8
        want = min(self.max_tokens, room)
        if want < 64:
            self.stats["ctx_exhausted"] += 1
            return {"error": f"context exhausted: {n_prompt} of {self.ctx_len} tokens"}
        if want < self.max_tokens:
            self.stats["ctx_clamped"] += 1
        payload = {
            "model": self.model, "prompt": prompt,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": want, "skip_special_tokens": False,
        }
        if seed is not None:
            payload["seed"] = seed
        if stop is not None:
            payload["stop"] = stop
        req = urllib.request.Request(
            base.rstrip("/") + "/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    self.stats["ok"] += 1
                    return json.loads(resp.read())
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                self.stats["retry"] += 1
                if attempt == 2:
                    self.stats["fail"] += 1
                    return {"error": str(exc)}
                time.sleep(2 * (attempt + 1))
        return {"error": "unreachable"}

    def complete_turn(self, prompt: str, *, temperature: float | None = None,
                      seed: int | None = None) -> dict:
        """One turn, optionally as the TWO calls training actually encodes.

        `board_harmony.py` closes the analysis channel with a supervised `<|end|>`
        and then emits `<|start|>assistant` with ROLE_IGNORE, so the label at the
        position where the model has just produced `<|end|>` is IGNORE_INDEX.
        Nothing in the loss says what follows `<|end|>`; its docstring says the
        harness supplies it. In one call this harness does not, and the model has
        to guess an unscored position, which a trained checkpoint may or may not
        get right. Supplying it removes the guess. The merged text is the shape the
        one-call parser already expects, so nothing downstream changes.
        """
        if not self.two_stage or _NO_ANALYSIS[0]:
            return self.complete(prompt, temperature=temperature, seed=seed)
        first = self.complete(prompt, temperature=temperature, seed=seed,
                              stop=[ANALYSIS_END])
        if "error" in first:
            return first
        c0 = (first.get("choices") or [{}])[0]
        think = c0.get("text") or ""
        # Out of tokens before the channel ever closed is a real truncation, not a
        # boundary miss: hand it back untouched so the finish_reason tally sees it.
        if c0.get("finish_reason") == "length":
            self.stats["stage1_length"] += 1
            return first
        second = self.complete(prompt + think + ANALYSIS_END + ASSISTANT_TURN,
                               temperature=temperature, seed=seed)
        if "error" in second:
            return second
        c1 = (second.get("choices") or [{}])[0]
        self.stats["two_stage"] += 1
        merged = dict(second)
        merged["choices"] = [dict(c1, text=think + ANALYSIS_END + ASSISTANT_TURN
                                  + (c1.get("text") or ""))]
        u0 = (first.get("usage") or {}).get("completion_tokens") or 0
        u1 = (second.get("usage") or {}).get("completion_tokens") or 0
        merged["usage"] = dict(second.get("usage") or {}, completion_tokens=u0 + u1)
        return merged


_NO_ANALYSIS = [False]


def build_prompt(developer: str, tools: list[dict], harmony_msgs: list[dict],
                 reasoning_effort: str = "medium") -> str:
    """The exact training-time prefix, rendered through openai_harmony -- not a
    regenerated chat template -- plus the forced analysis-channel opener.

    `reasoning_effort` reaches the SYSTEM message and has to match the corpus the
    checkpoint was trained on. Without it `verify_harmony.build` falls back to `low`
    and prints a warning: the student then reads `Reasoning: low` where it was
    trained under `Reasoning: medium`, which is a prompt it never saw. The route-loop
    corpora stamp medium, so that is the default here.
    """
    row = {"messages": [{"role": "developer", "content": developer}],
           "tools": tools, "harmony_messages": harmony_msgs,
           "reasoning_effort": reasoning_effort}
    convo = V.build(row)
    toks = ENC.render_conversation_for_completion(convo, Role.ASSISTANT)
    # A no-reasoning arm never produced an analysis channel, so forcing one asks it
    # for a shape it has not got. Left to itself from `<|start|>assistant` it emits
    # ` to=functions.board_act<|channel|>commentary json<|message|>{...}` -- which is
    # what its corpus holds.
    return ENC.decode(toks) + ("" if _NO_ANALYSIS[0] else FORCED_PREFIX)


def parse_forced_completion(text: str) -> tuple[str, Optional[str], Optional[str]]:
    """Split one forced-reasoning completion into (reasoning, action_json, kind).

    `kind` is "commentary" (a tool call followed), "final" (the model handed
    over instead), or None (neither marker showed up -- malformed, out of
    tokens, or something else broke the expected shape).
    """
    end_at = text.find(ANALYSIS_END)
    if _NO_ANALYSIS[0] and end_at < 0:
        # nothing was forced open, so there is no reasoning to split off and the
        # commentary marker is in the completion itself
        reasoning, rest = "", text
    else:
        reasoning = text[:end_at].strip() if end_at >= 0 else text.strip()
        rest = text[end_at + len(ANALYSIS_END):] if end_at >= 0 else ""
    call_at = rest.find(CALL_MARK)
    final_at = rest.find(FINAL_MARK)
    if call_at >= 0 and (final_at < 0 or call_at < final_at):
        msg_at = rest.find(MSG_MARK, call_at)
        if msg_at < 0:
            return reasoning, None, None
        payload = rest[msg_at + len(MSG_MARK):].strip()
        # A trailing special token (<|call|>/<|return|>) may or may not survive
        # in `text` depending on whether it matched a stop sequence; strip any
        # leftover marker rather than feeding it to the JSON parser.
        cut = payload.find("<|")
        if cut >= 0:
            payload = payload[:cut]
        return reasoning, payload.strip(), "commentary"
    if final_at >= 0:
        msg_at = rest.find(MSG_MARK, final_at)
        content = rest[msg_at + len(MSG_MARK):].strip() if msg_at >= 0 else ""
        cut = content.find("<|")
        if cut >= 0:
            content = content[:cut]
        return reasoning, content, "final"
    return reasoning, None, None


def run_target(target: str, world: LiveWorld, client: RawClient,
               style: R.RenderStyle, *, budget: int, max_depth: int,
               max_turns: int, developer: str, rollout: int = 0) -> dict:
    board = Board(target, world, max_depth=max_depth, budget=budget)
    t_start = time.time()
    t_first_route: Optional[float] = None
    first_solved_calls: Optional[int] = None
    output: list[dict] = []
    harmony_msgs: list[dict] = []
    tools = [{"name": H.TOOL_NAME, "description": H.TOOL_DESC, "parameters": H.ACT_SCHEMA}]
    errors: Counter[str] = Counter()
    illegal_streak = 0            # used only in consecutive mode (reset to 0 on success)
    stop = "max_turns"

    obs = R.render_env(board, style)
    harmony_msgs.append({"role": "user", "content": obs})

    turn = 0
    for turn in range(max_turns):
        if turn == 0:
            raw = {"turn": 0, "reasoning": None,
                   "arguments": json.dumps({"actions": [{"type": "open", "mid": board.root}]}),
                   "kind": "commentary", "finish_reason": "skipped_root_open", "usage": 0}
            output.append(raw)
            harmony_msgs.append({"role": "assistant", "channel": "commentary",
                                 "recipient": f"functions.{H.TOOL_NAME}", "content_type": "json",
                                 "content": raw["arguments"], "supervised": True})
            board.apply(parse_act_json(raw["arguments"]))
            obs = R.render_env(board, style)
            harmony_msgs.append({"role": "tool", "name": f"functions.{H.TOOL_NAME}",
                                 "channel": "commentary", "content": obs})
            continue
        needs_open = ([m for m in board.open_mols() if board.mols[m].menu is None]
                      if _SKIP_OPEN_REASONING[0] else [])
        if needs_open:
            mid = needs_open[0]
            raw = {"turn": turn, "reasoning": None,
                   "arguments": json.dumps({"actions": [{"type": "open", "mid": mid}]}),
                   "kind": "commentary", "finish_reason": "skipped_deterministic_open", "usage": 0}
            output.append(raw)
            harmony_msgs.append({"role": "assistant", "channel": "commentary",
                                 "recipient": f"functions.{H.TOOL_NAME}", "content_type": "json",
                                 "content": raw["arguments"], "supervised": True})
            try:
                board.apply(parse_act_json(raw["arguments"]))
            except NoMenu as exc:
                errors["no_menu"] += 1
                rej = f"rejected: {exc}"
                raw["rejected"] = rej
                harmony_msgs.append({"role": "tool", "name": f"functions.{H.TOOL_NAME}",
                                     "channel": "commentary", "content": rej})
                continue
            except BoardError as exc:
                errors["illegal_action"] += 1
                rej = f"rejected: {exc}"
                raw["rejected"] = rej
                errors[f"illegal:{str(exc).split(' ', 1)[-1][:40]}"] += 1
                harmony_msgs.append({"role": "tool", "name": f"functions.{H.TOOL_NAME}",
                                     "channel": "commentary", "content": rej})
                illegal_streak += 1
                if (illegal_streak if _ILLEGAL_STREAK[0]
                        else errors["illegal_action"]) >= _ILLEGAL_CAP[0]:
                    stop = "illegal_action"
                    break
                continue
            illegal_streak = 0        # success, so clear the consecutive-failure record
            if t_first_route is None and board.routes:
                t_first_route = time.time() - t_start
            if first_solved_calls is None and board.solved():
                first_solved_calls = board.budget_used
            obs = R.render_env(board, style)
            harmony_msgs.append({"role": "tool", "name": f"functions.{H.TOOL_NAME}",
                                 "channel": "commentary", "content": obs})
            if board.budget_used >= budget:
                stop = "budget"
                break
            if not board.open_mols() and not board.solved():
                stop = "stuck"
                break
            continue
        prompt = build_prompt(developer, tools, harmony_msgs, _EFFORT[0])
        reply = client.complete_turn(prompt, seed=_SEED_BASE[0] + rollout)
        if "error" in reply:
            errors["model_error"] += 1
            stop = "model_error"
            break
        choice = (reply.get("choices") or [{}])[0]
        text = choice.get("text") or ""
        reasoning, payload, kind = parse_forced_completion(text)
        raw = {
            "turn": turn, "reasoning": reasoning, "arguments": payload,
            "kind": kind, "finish_reason": choice.get("finish_reason"),
            "usage": (reply.get("usage") or {}).get("completion_tokens"),
        }
        output.append(raw)

        if kind == "final" or (kind is None and payload is None):
            stop = "final" if kind == "final" else "unparsed_completion"
            if reasoning:
                harmony_msgs.append({"role": "assistant", "channel": "analysis",
                                     "content": reasoning, "supervised": True})
            if kind == "final":
                harmony_msgs.append({"role": "assistant", "channel": "final",
                                     "content": payload or "", "supervised": True})
            break

        harmony_msgs.append({"role": "assistant", "channel": "analysis",
                             "content": reasoning, "supervised": True})
        harmony_msgs.append({"role": "assistant", "channel": "commentary",
                             "recipient": f"functions.{H.TOOL_NAME}",
                             "content_type": "json", "content": payload,
                             "supervised": True})
        try:
            actions = parse_act_json(payload)
        except ActError as exc:
            errors["unparsed_call"] += 1
            rej = f"rejected: {exc}"
            raw["rejected"] = rej
            harmony_msgs.append({"role": "tool", "name": f"functions.{H.TOOL_NAME}",
                                 "channel": "commentary", "content": rej})
            if errors["unparsed_call"] >= 3:
                stop = "unparsed_call"
                break
            continue
        if _MAX_OPENS[0]:
            kept, seen_open = [], 0
            for a in actions:
                if isinstance(a, Open):
                    seen_open += 1
                    if seen_open > _MAX_OPENS[0]:
                        errors["open_truncated"] += 1
                        continue
                kept.append(a)
            actions = kept
            if not actions:
                continue
        if any(isinstance(a, Terminate) for a in actions):
            stop = "terminate"
            break
        try:
            board.apply(actions)
        except NoMenu as exc:
            errors["no_menu"] += 1
            rej = f"rejected: {exc}"
            raw["rejected"] = rej
            harmony_msgs.append({"role": "tool", "name": f"functions.{H.TOOL_NAME}",
                                 "channel": "commentary", "content": rej})
            continue
        except BoardError as exc:
            errors["illegal_action"] += 1
            rej = f"rejected: {exc}"
            raw["rejected"] = rej
            errors[f"illegal:{str(exc).split(' ', 1)[-1][:40]}"] += 1
            harmony_msgs.append({"role": "tool", "name": f"functions.{H.TOOL_NAME}",
                                 "channel": "commentary", "content": rej})
            illegal_streak += 1
            if (illegal_streak if _ILLEGAL_STREAK[0]
                    else errors["illegal_action"]) >= _ILLEGAL_CAP[0]:
                stop = "illegal_action"
                break
            continue
        illegal_streak = 0            # success, so clear the consecutive-failure record
        if t_first_route is None and board.routes:
            t_first_route = time.time() - t_start
        if first_solved_calls is None and board.solved():
            first_solved_calls = board.budget_used
        obs = R.render_env(board, style)
        harmony_msgs.append({"role": "tool", "name": f"functions.{H.TOOL_NAME}",
                             "channel": "commentary", "content": obs})
        if _STOP_EP_ON_SOLVE[0] and board.solved():
            stop = "solved"
            break
        if board.budget_used >= budget:
            stop = "budget"
            break
        if not board.open_mols() and not board.solved():
            stop = "stuck"
            break

    implicit = False
    if board.solved() and not board.routes:
        via = board.mols[board.root].closed_via
        if via and via != "stock":
            implicit = bool(board._register(board._choices_from(via), via))

    routes = []
    for route in board.routes:
        rxns = R._route_rxns(board, route.root_rxn)
        vals = [R.signals_of_rxn(board.rxns[r], style)[1] for r in rxns]
        vals = [v for v in vals if v is not None]
        leaves = R._route_leaves(board, route.root_rxn)
        priced = [board.mols[m].price_ln for m in leaves if board.mols[m].price_ln is not None]
        routes.append({
            "label": route.label, "n_steps": len(rxns),
            "weakest": min(vals) if vals else None,
            "n_leaves": len(leaves), "n_unpriced": len(leaves) - len(priced),
            "cost_usd": round(sum(math.exp(v) for v in priced), 2) if priced else None,
            "steps": [[board.mols[board.rxns[r].parent].smiles,
                       [board.mols[p].smiles for p in board.rxns[r].pieces]]
                      for r in rxns],
        })

    return {
        "target": target, "rollout": rollout, "stock": world.stock_name,
        "solved": board.solved(), "stop": stop,
        "turns": turn + 1, "calls": board.budget_used,
        "wall_s": round(time.time() - t_start, 2),
        "first_route_s": round(t_first_route, 2) if t_first_route is not None else None,
        "first_solved_calls": first_solved_calls,
        "n_open_left": len(board.open_mols()), "n_dead": len(board.dead_mols()),
        "routes": routes, "routes_implicit": implicit,
        "errors": dict(errors), "output": output,
        "harmony_messages": harmony_msgs, "forced_reasoning": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--model-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--menu-url", required=True)
    ap.add_argument("--stock", default="emols")
    ap.add_argument("--reasoning", default="medium", choices=["low", "medium", "high"],
                    help="the reasoning_effort the SYSTEM message states. Must match the "
                         "corpus the checkpoint was trained on -- the route-loop corpora "
                         "stamp medium; serving low is a prompt the student never saw")
    ap.add_argument("--illegal-cap", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy, the setting a solve rate is comparable at. "
                         "Raise it for the iterative protocol so a restart explores "
                         "instead of replaying the same trajectory.")
    ap.add_argument("--stop-episode-on-solve", action="store_true",
                    help="End the episode at the solve rather than at the illegal cap. "
                         "For the restart protocol: the post-solve tail buys nothing a "
                         "restart would not.")
    ap.add_argument("--max-opens-per-turn", type=int, default=0,
                    help="0 = unlimited (trained behaviour). 1 serialises opens, the "
                         "ablation arm for whether batching opens actually saves calls.")
    ap.add_argument("--seed-base", type=int, default=1234,
                    help="per-rollout seed = seed-base + rollout index")
    ap.add_argument("--price-live", action="store_true",
                    help="price a buyable molecule the cache does not carry with "
                         "MolPrice instead of rendering it with no `$`. See the same "
                         "flag on eval_board_agent.py. CPU only.")
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=10)
    # 120, not 60. A large unique-call budget cannot be spent inside 60 turns, so a budget
    # curve drawn from such runs would be a turn-cap curve wearing a budget label.
    ap.add_argument("--max-turns", type=int, default=120)
    ap.add_argument("--menu-show", type=int, default=10)
    ap.add_argument("--signals", default="q,p,rt")
    ap.add_argument("--deciding", default="p")
    ap.add_argument("--handover", default="flow")
    ap.add_argument("--cutoff", type=float, default=0.05)
    # 3000 max new tokens per turn. The checkpoint can write longer analysis at eval than
    # the corpus holds, and a clipped decode cuts the reasoning mid-sentence so the tool
    # call never follows: it spends a turn and produces no action, which reads as a weak
    # model rather than as a truncation.
    ap.add_argument("--max-tokens", type=int, default=3000,
                    help="reasoning + tool call together; the schema-form "
                         "reasoning needs headroom past a normal tool-call-only "
                         "budget")
    ap.add_argument("--no-analysis", action="store_true",
                    help="do not force the analysis channel open. Required for a "
                         "corpus rendered with `--analysis none` (the ablation "
                         "arms): those checkpoints go straight to the tool call, "
                         "and a forced `<|channel|>analysis<|message|>` asks for a "
                         "channel they never produced. Implies single-call turns.")
    ap.add_argument("--ctx-len", type=int, default=131072,
                    help="the served --max-model-len. The client sizes each ask to "
                         "the room left in it; set it too high and vllm 400s, too "
                         "low and long episodes are cut short.")
    ap.add_argument("--two-stage", action="store_true",
                    help="close the analysis channel with a stop at `<|end|>` and "
                         "supply `<|start|>assistant` before asking for the tool "
                         "call -- the split training encodes but a single call "
                         "does not. Default off (single call).")
    ap.add_argument("--developer-file", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("RP_WORKERS", "8")),
                    help="each call decodes the reasoning as well as the tool "
                         "call, so each worker holds a replica longer")
    ap.add_argument("--limit", type=int, default=0)
    # Same flag as eval_board_agent.py; this script reuses its LiveWorld but not its parser,
    # so without it rt comes from the recorded cache alone and render.py shows `?` for the
    # rest, where the SFT corpus carries rt on every candidate.
    ap.add_argument("--rt", default=None,
                    help="live:URL[,URL...] to score round-trip on the menu; "
                         "default is the recorded cache only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-open-reasoning", dest="skip_open_reasoning",
                    action="store_true", default=False,
                    help="deterministically `open` a piece with no menu yet, in "
                         "first-seen order, instead of asking the model. Off by "
                         "default: see --no-skip-open-reasoning. Turn 0 is "
                         "skipped regardless of this flag.")
    ap.add_argument("--no-skip-open-reasoning", dest="skip_open_reasoning",
                    action="store_false",
                    help="the default. The model chooses what to open, which is the "
                         "behaviour the multi-route claim is about, and skipping "
                         "removes it entirely: with the skip on, the harness issues "
                         "the opens, so the model never expands more than one piece "
                         "a turn whatever its corpus does. Under --illegal-cap 1 one "
                         "hallucinated menu ends the episode; with a consecutive "
                         "cap the model recovers and chooses its own opens.")
    args = ap.parse_args()
    _ILLEGAL_CAP[0] = args.illegal_cap
    _SEED_BASE[0] = args.seed_base
    _NO_ANALYSIS[0] = args.no_analysis
    _MAX_OPENS[0] = args.max_opens_per_turn
    _STOP_EP_ON_SOLVE[0] = args.stop_episode_on_solve
    _EFFORT[0] = args.reasoning
    _SKIP_OPEN_REASONING[0] = args.skip_open_reasoning
    developer = Path(args.developer_file).read_text()

    targets = []
    with open(args.targets) as fh:
        for line in fh:
            row = json.loads(line)
            targets.append(row["target"] if isinstance(row, dict) else row)
    if args.limit:
        targets = targets[: args.limit]

    scores = load_node_scores(want=("rt", "price", "plaus"))
    stock = C.Stock(args.stock)
    print(f"# stock {args.stock}: {len(stock):,} keys ({stock.mode})", file=sys.stderr)
    print(f"# forced-reasoning eval: max_tokens={args.max_tokens}"
          f"{' two-stage' if args.two_stage else ''}", file=sys.stderr)

    rt_url = args.rt.split("live:", 1)[1] if (args.rt or "").startswith("live:") else None
    if rt_url is None:
        print("# WARNING: --rt not given -- round-trip is cache-only and every miss "
              "renders as `?`, where the training corpus carries rt on every candidate.",
              flush=True)
    else:
        print(f"# rt: live on cache miss, {len(rt_url.split(','))} forward replica(s)", flush=True)
    world = LiveWorld(
        [u.strip() for u in args.menu_url.split(",") if u.strip()],
        stock=stock, stock_name=args.stock, prices=scores.price, rt_cache=scores.rt,
        plaus_cache=scores.plaus, top_k=args.menu_show, rt_url=rt_url,
        price_live=args.price_live,
    )
    client = RawClient([u.strip() for u in args.model_url.split(",") if u.strip()],
                       args.model, max_tokens=args.max_tokens,
                       temperature=args.temperature, two_stage=args.two_stage,
                       ctx_len=args.ctx_len)
    style = R.RenderStyle(
        menu_show=args.menu_show, menu_cutoff=args.cutoff,
        signal_order=tuple(args.signals.split(",")),
        cutoff_signal=args.deciding, route_signal=args.deciding,
        handover=args.handover,
    )

    done = 0
    tally: Counter[str] = Counter()
    lock = threading.Lock()
    out_f = open(args.out, "w")

    def work(target):
        try:
            return run_target(target, world, client, style, budget=args.budget,
                              max_depth=args.max_depth, max_turns=args.max_turns,
                              developer=developer)
        except Exception as exc:                       # noqa: BLE001
            return {"target": target, "solved": False, "stop": "crash",
                    "error": f"{type(exc).__name__}: {exc}",
                    "routes": [], "errors": {}, "turns": 0, "calls": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, t) for t in targets]
        for fut in as_completed(futures):
            row = fut.result()
            with lock:
                done += 1
                tally["solved"] += bool(row["solved"])
                tally["calls"] += row.get("calls", 0)
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                if done % 25 == 0 or done == len(targets):
                    print(f"# {done}/{len(targets)} solved={tally['solved']} "
                          f"({tally['solved']/done:.1%}) "
                          f"mean_calls={tally['calls']/done:.1f}", file=sys.stderr)
    out_f.close()
    print(f"# client: {dict(client.stats)}", file=sys.stderr)
    print(f"# wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
