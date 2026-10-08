#!/usr/bin/env python
"""Closed-loop eval: the model plays the board, live, against real servers.

Nothing here is replayed.  The menu comes from a single-step server, the
plausibility from AiZynthFinder's filter policy (a local ONNX model), buyability from the same stock the training corpus used, and the price
from MolPrice's table.  The model is served over an OpenAI-compatible endpoint and
answers with a `board_act` tool call, which the board applies.  So a solve here is
a solve the environment agrees with, not a string match against a recorded route.

The observation has to be the one the model was trained on, which is why the
developer message and the tool schema come from `board.harmony` rather than being
restated: they are the same objects the dataset was built with.

    python "$RP_BOARD"/eval_board_agent.py \\
        --targets "$RP_PROTOCOL"/targets_uspto190.jsonl \\
        --model-url http://127.0.0.1:$RP_PORT_LLM/v1 --model retroplanner \\
        --menu-url http://127.0.0.1:$RP_PORT_MENU/predict \\
        --developer-file "$RP_PROTOCOL"/developer/dev_retroplanner.txt \\
        --stock emols --budget 300 --max-depth 10 --workers $RP_WORKERS \\
        --rt live:http://127.0.0.1:$RP_PORT_FORWARD \\
        --out results/board_eval_uspto190.jsonl

`--rt live:URL` puts round-trip on the menu; without it rt is read from the
recorded cache and rendered as `rt—` where it is absent, which is a distribution
shift from training, so the flag is worth the server.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import os
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
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

from board import harmony as H          # noqa: E402
from board import render as R           # noqa: E402
from board.episode import load_node_scores  # noqa: E402
from board.parse import ActError, parse_act_json  # noqa: E402
from board.state import (Board, BoardError, Candidate, Done, NoMenu,
                         Terminate)  # noqa: E402

import traj_route_common as C           # noqa: E402


# ------------------------------------------------------------------ wire text
class WireText:
    """Renders an episode to text, under the SAME field names the corpus uses.

    The training corpus writes two renderings of every episode and the names are
    not interchangeable:

      * `wire_text` -- openai_harmony's rendering, which is what vLLM actually
        feeds gpt-oss and therefore what the checkpoint was trained on;
      * `raw_text` -- the HF `chat_template.jinja` rendering, which differs in
        small ways (it stamps `Current date`, it renders the tool schema through
        a different path) and is kept as the cross-check.

    Naming the harmony render `raw_text` here would make an eval dump look
    comparable to the corpus while silently comparing harmony against jinja, so
    both names mean here exactly what they mean there.  The renderers are
    verify_harmony's and board.harmony's, not copies.  Missing dependencies leave
    the fields out rather than faking them; `verify_harmony.py <dump> --augment`
    fills `wire_*` in afterwards.
    """

    def __init__(self, style: R.RenderStyle, chat_template: bool = True) -> None:
        self.enc = self.build = self.chat = None
        self.reason: Optional[str] = None
        try:
            from openai_harmony import HarmonyEncodingName, load_harmony_encoding
            import verify_harmony as V
            self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            self.build, self._V = V.build, V
        except Exception as exc:                      # pragma: no cover
            self.reason = f"{type(exc).__name__}: {exc}"
        if chat_template:
            try:
                self.chat = H.ChatRenderer()
            except Exception as exc:                  # pragma: no cover
                self.chat_reason = f"{type(exc).__name__}: {exc}"

    @property
    def ok(self) -> bool:
        return self.enc is not None or self.chat is not None

    def __call__(self, row: dict) -> dict:
        got: dict = {}
        if self.enc is not None:
            toks = self.enc.render_conversation(self.build(row))
            text = self.enc.decode(toks)
            sysm = self._V.SYS_RE.search(text)
            devm = self._V.DEV_RE.search(text)
            got["wire_text"] = text
            got["n_tokens"] = len(toks)
            if sysm and devm:
                got["wire_system"] = sysm.group(1)
                got["wire_developer"] = devm.group(1)
        if self.chat is not None:
            got["raw_text"] = self.chat.render(
                row["messages"], row.get("tools", []),
                reasoning_effort=row.get("reasoning_effort", "low"))
        return got


# --------------------------------------------------------------------- world
class LiveWorld:
    """The board's environment, wired to live models instead of a cache.

    Thread-safe by lock, not by luck: one instance serves every target in the
    run so the menu cache is shared, and the ONNX session underneath the
    plausibility scorer is not reentrant.
    """

    def __init__(self, menu_urls: list[str], stock, prices: dict,
                 rt_cache: dict, plaus_cache: dict, top_k: int = 10,
                 rt_url: Optional[str] = None, timeout: float = 240.0,
                 stock_name: str = "?", price_live: bool = False):
        self._urls = cycle(menu_urls)
        # Which buyability convention this run used.  It rides on the row because a
        # solve rate is meaningless without it: a route closed under one stock need
        # not close under another.
        self.stock_name = stock_name
        self._url_lock = threading.Lock()
        self.stock = stock
        self.prices = prices
        self.rt_cache = rt_cache
        self.plaus_cache = plaus_cache
        self.rt_url = rt_url
        # MolPrice for molecules the cache does not carry.  Off by default because it
        # changes the cost axis, and a run that prices live is not comparable with one
        # that leaves the gaps empty.  Pure numpy, so it costs CPU and no GPU.
        self.price_live = price_live
        # Live round-trip values are persisted, not only kept in the in-memory cache: the
        # metrics scripts score the same reactions afterwards, and without this they would
        # have to recompute every one. New values accumulate in `_rt_new` and are flushed to
        # a per-process overlay that `feas_cost.RtScorer` and the metrics scripts glob.
        #
        # ONE FILE PER PROCESS: the flush is tmp + os.replace, so shards sharing one path
        # delete each other's tmp and die with FileNotFoundError. RT_OVERLAY_TAG names the
        # run; the pid keeps its shards apart.
        self._rt_new: dict = {}
        self._rt_dirty = 0
        tag = os.environ.get("RT_OVERLAY_TAG", "board_eval")
        self._rt_overlay = (Path(os.environ.get("RT_OVERLAY_DIR",
                                                "data/route_search/feas"))
                            / f"rt_overlay_{tag}.pid{os.getpid()}.json")
        # Fleet rotation. roundtrip()'s own comma handling is bypassed by url_override,
        # so without these a fleet of replicas is either one replica or an unparseable URL.
        rt_conc = int(os.environ.get("RT_CONCURRENCY", "3"))
        self._rt_urls = [u.strip() for u in (rt_url or "").split(",") if u.strip()]
        self._rt_sem = threading.Semaphore(max(1, rt_conc * max(1, len(self._rt_urls))))
        self._rt_rr = 0
        self._rt_rr_lock = threading.Lock()
        self.top_k = top_k
        # Number of candidates to hold in the cache. Larger than top_k only when the filter
        # is on.
        self._pool_n = max(top_k, int(os.environ.get("BOARD_MENU_POOL", str(top_k))))
        self.timeout = timeout
        self.menus: dict[str, list[tuple[list[str], float]]] = {}
        self._menu_lock = threading.Lock()
        self._score_lock = threading.Lock()
        self.stats = Counter()

    # -- q ---------------------------------------------------------------
    def _fetch(self, smiles: str, max_attempts: int = 50):
        # A dropped/slow menu reply must not come back as `[]` on the FIRST
        # failure -- that is indistinguishable from "this molecule genuinely has no
        # candidates" (NoMenu), which silently records a fake dead end
        # (what a starved single-step pool produces). Retry round-robin across every replica, with backoff, rather than
        # ever answering "no menu" because one request happened to fail.
        # If BOARD_MENU_POOL is larger than top_k, fetch and cache that many. Headroom so the
        # board can still fill the screen to top_k after filtering out cycle candidates --
        # same structure as RetroAgent's tools/ml_retro.py taking max(top_k,50) and keeping
        # top_k after filtering. The default is top_k, which fetches exactly top_k.
        body = json.dumps({"smiles": smiles, "top_n": self._pool_n}).encode()
        last_exc: Optional[BaseException] = None
        # An HTTP 200 carrying `[]` is retried too, not only transport errors and error
        # objects: R-SMILES (root_aligned) can return an empty list for a molecule it answers
        # on the next call, when the augmented beam draws happen to leave no valid precursor.
        # So the same molecule can be empty or not per CALL, and one empty reply must not
        # become a fake dead end that ends the whole target.
        # RETRY EMPTY with a short backoff, and cap it: without a cap a molecule that really
        # has no candidates would ride the full attempt path into a long stall and a
        # RuntimeError.
        empty_retry_max = 6
        empty_tries = 0
        for attempt in range(max_attempts):
            with self._url_lock:
                url = next(self._urls)
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    rows = json.loads(resp.read())
            except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
                last_exc = exc
                self.stats["menu_retry"] += 1
                time.sleep(min(2.0 * (attempt + 1), 20.0))
                continue
            if isinstance(rows, dict):
                last_exc = ValueError(f"menu server returned an error object: {rows}")
                self.stats["menu_retry"] += 1
                time.sleep(min(2.0 * (attempt + 1), 20.0))
                continue
            out = []
            for r in rows[: self._pool_n]:
                pre = r.get("precursors") or r.get("reactants")
                if pre:
                    out.append((list(pre), float(r.get("confidence", 0.0))))
            if not out and empty_tries < empty_retry_max:
                empty_tries += 1
                self.stats["menu_empty_retry"] += 1
                time.sleep(0.5 * empty_tries)
                continue
            if not out:
                self.stats["menu_empty_final"] += 1
            self.stats["menu_call"] += 1
            return out
        # Every attempt failed: this is a real, not transient, problem --
        # surface it loudly instead of silently reporting "no candidates".
        self.stats["menu_error"] += 1
        raise RuntimeError(
            f"menu fetch for {smiles!r} failed after {max_attempts} attempts: {last_exc}")

    # -- p and rt --------------------------------------------------------
    def _plaus(self, product: str, menu) -> list[Optional[float]]:
        """One ONNX batch per menu, cache first.  The filter is the axis that
        decides, so a missing value is left as None rather than defaulted."""
        keys = [f"{product}>>{'.'.join(r)}" for r, _ in menu]
        out: list[Optional[float]] = [self.plaus_cache.get(k) for k in keys]
        todo = [i for i, v in enumerate(out) if v is None]
        if todo:
            # score_reactions takes (product, [reactants]) pairs, not rxn keys --
            # the keys are only how the cache is addressed.
            from reaction_mcp.scoring.plausibility import score_reactions
            pairs = [(product, list(menu[i][0])) for i in todo]
            # NO LOCK. ort.InferenceSession.run is thread-safe; a race on the lazily
            # initialized _sess only builds the session twice, not an error; and the _canon/_fp
            # caches are atomic under the GIL, so the only cost is duplicate computation. Holding
            # the lock, by contrast, makes every worker in the process wait on a single scoring
            # call.
            if True:
                try:
                    scored = score_reactions(pairs)
                except Exception as exc:                   # noqa: BLE001
                    self.stats["plaus_error"] += len(todo)
                    self.stats[f"plaus_exc:{type(exc).__name__}"] += 1
                    scored = [None] * len(todo)
            for i, v in zip(todo, scored):
                if v is not None:
                    out[i] = float(v)
                    self.plaus_cache[keys[i]] = float(v)
            self.stats["plaus_scored"] += len(todo)
        return out

    def _rt(self, product: str, menu) -> list[Optional[float]]:
        # Two shapes, deliberately: the CACHE is keyed by the reaction string the
        # recorded caches use, and `roundtrip` takes (product, [reactants]) pairs.
        # Passing the key string to the scorer raises inside a list comprehension
        # -- "too many values to unpack" -- which the except turns into rt_error
        # for every candidate, so the axis silently reads as "never recovered".
        keys = [f"{product}>>{'.'.join(r)}" for r, _ in menu]
        pairs = [(product, list(r)) for r, _ in menu]
        out = [self.rt_cache.get(k) for k in keys]
        if self.rt_url is None:
            self.stats["rt_from_cache"] += sum(1 for v in out if v is not None)
            self.stats["rt_absent"] += sum(1 for v in out if v is None)
            return out
        todo = [i for i, v in enumerate(out) if v is None and keys[i] not in self.rt_cache]
        if todo:
            from reaction_mcp.scoring.roundtrip import roundtrip
            # NO LOCK. A multi-second remote call, and `_ask` documents itself as driven by many
            # threads at once (it holds its own round-robin lock). Holding `_score_lock` for this
            # call ties the whole run to one round-trip batch at a time, however many workers or
            # replicas there are.
            if True:
                try:
                    # `url_override` bypasses roundtrip()'s own comma-list rotation, so the
                    # rotation has to happen HERE or the whole list is handed over as one URL and
                    # every request dies with "Failed to parse".
                    if len(self._rt_urls) > 1:
                        with self._rt_rr_lock:
                            self._rt_rr = (self._rt_rr + 1) % len(self._rt_urls)
                            endpoint = self._rt_urls[self._rt_rr]
                    else:
                        endpoint = self.rt_url
                    got = roundtrip([pairs[i] for i in todo], top_k=5,
                                    url_override=endpoint)
                except Exception as exc:
                    self.stats["rt_error"] += len(todo)
                    self.stats[f"rt_error:{type(exc).__name__}"] += 1
                    got = [None] * len(todo)
            for i, v in zip(todo, got):
                self.rt_cache[keys[i]] = v
                self._rt_new[keys[i]] = v
                out[i] = v
            self.stats["rt_scored"] += len(todo)
            self._rt_dirty += len(todo)
            # Scaled interval, like DrawCache: the flush rewrites the whole overlay, so a fixed
            # interval is O(n^2) in bytes over a long run. A crash costs at most the unflushed
            # window.
            if self._rt_dirty >= max(200, len(self._rt_new) // 20):
                self._flush_rt()
        return out

    def _flush_rt(self) -> None:
        """Write the values THIS process scored. Only `_rt_new`, never the loaded cache --
        rewriting every inherited entry per shard would dwarf the new ones."""
        if not self._rt_new:
            return
        try:
            self._rt_overlay.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(self._rt_overlay) + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(self._rt_new, fh)
            os.replace(tmp, self._rt_overlay)
            self._rt_dirty = 0
            self.stats["rt_flushed"] = len(self._rt_new)
        except Exception as exc:                      # noqa: BLE001
            # A failed flush must not take the eval down with it: the values are still in
            # memory and the run's own results do not depend on them being on disk.
            self.stats[f"rt_flush_error:{type(exc).__name__}"] += 1

    # -- the World protocol ----------------------------------------------
    def menu(self, smiles: str, keep=None) -> list[Candidate]:
        """If `keep(reactants) -> bool` is given, filter with it, then keep only top_k.

        The cache (self.menus) holds the raw pool before filtering -- the predicate depends
        on board state, so it varies per call, not per molecule. Scoring (plaus/rt) runs
        only on the top_k left after filtering, so growing the pool does not raise scoring
        cost.
        """
        with self._menu_lock:
            rows = self.menus.get(smiles)
        if rows is None:
            rows = self._fetch(smiles)
            with self._menu_lock:
                self.menus[smiles] = rows
        if not rows:
            return []
        if keep is not None:
            kept = [r for r in rows if keep(r[0])]
            self.stats["menu_filtered"] += len(rows) - len(kept)
            if not kept:
                self.stats["menu_filtered_empty"] += 1
            rows = kept
        rows = rows[: self.top_k]
        if not rows:
            return []
        plaus = self._plaus(smiles, rows)
        rt = self._rt(smiles, rows)
        out = []
        for i, (reactants, q) in enumerate(rows):
            signals: dict = {"q": q, "rt": rt[i]}
            if plaus[i] is not None:
                signals["p"] = plaus[i]
            out.append(Candidate(i, list(reactants), signals))
        return out

    def _price(self, smiles: str) -> Optional[float]:
        """MolPrice for one molecule, ln(USD/mmol), or None.

        `predict_price` returns None, never 0.0, when MolPrice cannot build a fingerprint:
        on this scale 0.0 is a real price (1 USD/mmol), so a failure would otherwise enter
        the board as the cheapest thing on it.
        """
        from reaction_mcp.scoring.price import predict_price
        try:
            v = predict_price(smiles)
        except Exception as exc:                                       # noqa: BLE001
            self.stats["price_error"] += 1
            self.stats[f"price_exc:{type(exc).__name__}"] += 1
            return None
        self.stats["price_scored" if v is not None else "price_empty_fp"] += 1
        return v

    def info(self, smiles: str) -> tuple[bool, Optional[float]]:
        # Stock.has() projects to the stock's own key mode and memoises.
        buyable = bool(self.stock.has(smiles))
        if not buyable:
            return False, None
        p = self.prices.get(smiles)
        if p is not None:
            self.stats["price_from_cache"] += 1
            return True, float(p)
        if not self.price_live:
            self.stats["price_absent"] += 1
            return True, None
        with self._score_lock:
            p = self.prices.get(smiles)          # another thread may have filled it
            if p is None:
                p = self._price(smiles)
                if p is not None:
                    self.prices[smiles] = p
        return True, float(p) if p is not None else None


# --------------------------------------------------------------- the model
class RawCompletionClient:
    """The same contract as ChatClient, over /v1/completions on a harmony prefix.

    vllm's /chat/completions re-renders the conversation with the model's OWN chat
    template, and that is not the prefix these checkpoints were trained on: the chat
    path stamps a `Current date:` line into the system message and renders the tool
    schema differently from `verify_harmony.build`, which the corpus was rendered
    with. Served the chat rendering, a checkpoint can emit tokens the tool parser
    cannot read, and every episode then ends at turn 0 without an error.

    So the prompt is built here the way the corpus was, and the completion is parsed
    for the harmony call directly. The return shape is ChatClient's, so nothing
    downstream changes.
    """

    CALL_MARK = "<|channel|>commentary"
    FINAL_MARK = "<|channel|>final"
    MSG_MARK = "<|message|>"

    def __init__(self, urls: list[str], model: str, timeout: float = 600.0,
                 temperature: float = 0.0, max_tokens: int = 2048,
                 reasoning_effort: str = "low", ctx_len: int = 131072):
        self._urls = cycle(urls)
        self._lock = threading.Lock()
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.ctx_len = ctx_len
        self.stats = Counter()
        self._enc = None

    def _encoding(self):
        if self._enc is None:
            from openai_harmony import (HarmonyEncodingName,  # noqa: PLC0415
                                        load_harmony_encoding)
            self._enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        return self._enc

    def _prompt(self, messages: list[dict], tools: list[dict],
                harmony: list[dict]) -> str:
        """`harmony` is the driver's own harmony_msgs, not the OpenAI `messages`.

        The two lists are different shapes -- an assistant tool call is
        `{"role","channel","recipient","content"}` in one and `{"role","tool_calls"}`
        in the other -- and verify_harmony.build reads the first. Handing it the
        OpenAI list raises KeyError: 'content' on the first assistant turn.
        """
        from openai_harmony import Role                        # noqa: PLC0415
        import verify_harmony as V                             # noqa: PLC0415
        enc = self._encoding()
        dev = [m for m in messages if m.get("role") == "developer"]
        convo = V.build({"messages": dev, "tools": tools,
                         "harmony_messages": harmony,
                         "reasoning_effort": self.reasoning_effort})
        return enc.decode(enc.render_conversation_for_completion(convo, Role.ASSISTANT))

    def call(self, messages: list[dict], tools: list[dict],
             seed: Optional[int] = None, harmony: Optional[list[dict]] = None) -> dict:
        with self._lock:
            base = next(self._urls)
        prompt = self._prompt(messages, tools, harmony or [])
        room = self.ctx_len - len(self._encoding().encode(prompt, allowed_special="all")) - 8
        want = min(self.max_tokens, room)
        if want < 64:
            self.stats["ctx_exhausted"] += 1
            return {"error": "context exhausted", "http_status": 400}
        payload = {"model": self.model, "prompt": prompt, "max_tokens": want,
                   "temperature": self.temperature, "skip_special_tokens": False}
        if seed is not None:
            payload["seed"] = seed
        req = urllib.request.Request(
            base.rstrip("/") + "/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        for attempt in range(30):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    self.stats["ok"] += 1
                    return self._as_chat(json.loads(resp.read()))
            except urllib.error.HTTPError as exc:
                if 400 <= exc.code < 500:
                    self.stats["http_4xx"] += 1
                    return {"error": f"HTTP {exc.code}", "http_status": exc.code}
                self.stats["retry"] += 1
            except (urllib.error.URLError, TimeoutError, OSError):
                self.stats["retry"] += 1
            time.sleep(min(2 * (attempt + 1), 20))
        self.stats["fail"] += 1
        return {"error": "unreachable"}

    def _as_chat(self, reply: dict) -> dict:
        """Harmony completion -> the choices/message/tool_calls shape."""
        choice = (reply.get("choices") or [{}])[0]
        text = choice.get("text") or ""
        tool_calls, content = [], None
        call_at = text.find(self.CALL_MARK)
        final_at = text.find(self.FINAL_MARK)
        if call_at >= 0 and (final_at < 0 or call_at < final_at):
            msg_at = text.find(self.MSG_MARK, call_at)
            if msg_at >= 0:
                args = text[msg_at + len(self.MSG_MARK):]
                cut = args.find("<|")
                tool_calls = [{"type": "function", "function": {
                    "name": H.TOOL_NAME,
                    "arguments": (args[:cut] if cut >= 0 else args).strip()}}]
        elif final_at >= 0:
            msg_at = text.find(self.MSG_MARK, final_at)
            if msg_at >= 0:
                body = text[msg_at + len(self.MSG_MARK):]
                cut = body.find("<|")
                content = (body[:cut] if cut >= 0 else body).strip()
        return {"choices": [{"message": {"role": "assistant", "content": content,
                                         "tool_calls": tool_calls},
                             "finish_reason": choice.get("finish_reason")}],
                "usage": reply.get("usage") or {}}


class ChatClient:
    """One OpenAI-compatible endpoint, or several fronted round-robin."""

    def __init__(self, urls: list[str], model: str, timeout: float = 600.0,
                 temperature: float = 0.0, max_tokens: int = 2048,
                 reasoning_effort: str = "low"):
        self._urls = cycle(urls)
        self._lock = threading.Lock()
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.stats = Counter()

    def call(self, messages: list[dict], tools: list[dict],
             seed: Optional[int] = None, harmony: Optional[list[dict]] = None) -> dict:
        """`harmony` is accepted and ignored: the chat endpoint re-renders from
        `messages` itself. One signature so the driver does not branch."""
        with self._lock:
            base = next(self._urls)
        payload = {
            "model": self.model, "messages": messages,
            "tools": [{"type": "function", "function": t} for t in tools],
            "temperature": self.temperature, "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
        }
        if seed is not None:
            payload["seed"] = seed
        req = urllib.request.Request(
            base.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        # At high concurrency a transient failure here would end the whole
        # episode as "model_error" even though the server is healthy and the
        # NEXT attempt would succeed. Retry persistently instead of giving up
        # early -- a permanently-dead server still surfaces as "model_error"
        # after max_attempts, it just takes longer to get there.
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    self.stats["ok"] += 1
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                # A 4xx is the server rejecting THIS request as invalid, not a
                # transient hiccup -- an unbounded episode's conversation can
                # legitimately outgrow max-model-len (a 400 with "maximum
                # context length"), and retrying feeds it the exact same
                # oversized payload every time. Retrying achieves nothing but
                # burning up to max_attempts * 20s before failing anyway; fail
                # immediately and let run_target distinguish this from a real
                # server/network error.
                if 400 <= exc.code < 500:
                    self.stats["fail"] += 1
                    body = exc.read().decode(errors="replace")[:500]
                    return {"error": f"HTTP {exc.code}: {body}", "http_status": exc.code}
                # A 500 is USUALLY worth retrying (transient server trouble),
                # but vLLM's own openai_tool_parser crashes with an unhandled
                # JSONDecodeError when the model's tool-call JSON is malformed
                # (e.g. truncated at max_tokens) -- greedy decoding regenerates
                # the exact same malformed text every time, so retrying this
                # ONE specific 500 just burns max_attempts * 20s for nothing.
                body = exc.read().decode(errors="replace")
                if "JSONDecodeError" in body or "Error decoding JSON tool call" in body:
                    self.stats["fail"] += 1
                    return {"error": f"HTTP {exc.code}: {body[:500]}",
                            "http_status": exc.code, "malformed_tool_call": True}
                self.stats["retry"] += 1
                if attempt == max_attempts - 1:
                    self.stats["fail"] += 1
                    return {"error": f"HTTP {exc.code}: {body[:500]}"}
                time.sleep(min(2.0 * (attempt + 1), 20.0))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                self.stats["retry"] += 1
                if attempt == max_attempts - 1:
                    self.stats["fail"] += 1
                    return {"error": str(exc)}
                time.sleep(min(2.0 * (attempt + 1), 20.0))
        return {"error": "unreachable"}


# -------------------------------------------------------------- the episode
_ILLEGAL_CAP = [30]


def args_illegal_cap() -> int:
    """Rejections tolerated before the episode is abandoned.

    Every cycle attempt is a rejection, and the model attempts them -- the
    training corpus follows known-good routes, so it never contains a state where
    a candidate had to be refused.  A small cap abandons episodes the model would
    have recovered from.

    `--illegal-cap 1` is the NO-RETRY setting: the episode ends on the FIRST
    refused action. The default here is 30, a retry-tolerant setting.

    A cap that is too tight becomes the real limit on an episode instead of the
    call budget. The rejections are mostly state-tracking, not chemistry (ranking
    a molecule with no candidates on screen, or one with an unresolved reaction),
    so 30 gives the model room to recover from those; `--max-turns` remains the
    backstop against a model that only repeats itself.
    """
    return _ILLEGAL_CAP[0]


def run_target(target: str, world: LiveWorld, client: ChatClient,
               style: R.RenderStyle, *, budget: int, max_depth: int,
               max_turns: int, deciding: str, keep_transcript: bool,
               keep_messages: bool = True, wire: Optional["WireText"] = None,
               rollout: int = 0, seed: Optional[int] = None,
               reasoning_form: bool = False,
               developer_override: Optional[str] = None) -> dict:
    board = Board(target, world, max_depth=max_depth, budget=budget)
    t_start = time.time()
    t_first_route: Optional[float] = None
    first_route_calls: Optional[int] = None
    # The model's own output is always recorded: the arguments string it emitted,
    # any `content`, and any reasoning channel.  It is a few hundred bytes a turn
    # and it is the only thing in the dump that cannot be reconstructed -- without
    # it a run can be summarised but not read, and a rejection cannot be traced to
    # what the model actually wrote.  The BOARDS stay behind --transcripts because
    # those are kilobytes each and are reproducible from the actions.
    output: list[dict] = []
    # The conversation is kept in TWO shapes, the same two the training data uses:
    # `messages` is what was actually sent to the model (chat shape, with
    # tool_calls), and `harmony_messages` is the Harmony-native form that
    # verify_harmony.py can render to wire text and token spans.  Recording both
    # means an eval episode and a training episode are directly comparable, and a
    # divergence between what the model was trained on and what it was served
    # shows up as a diff rather than as a mystery.
    harmony_msgs: list[dict] = []
    # `developer_override` beats regenerating the instructions from the CURRENT
    # board/harmony.py: that module has drifted since some training corpora were
    # rendered (the analysis_form=True "How to think" section changed wording
    # and even which measures it lists), so a checkpoint trained on an older
    # render answers a regenerated prompt only approximately -- exactly the
    # silent-mismatch failure check_wire_match.py exists to catch. Pass the
    # corpus's own wire_developer verbatim instead of trusting the code to have
    # stood still.
    developer = developer_override or H.developer_instructions(
        style, deciding=deciding, analysis_form=reasoning_form)
    tools = [{"name": H.TOOL_NAME, "description": H.TOOL_DESC,
              "parameters": H.ACT_SCHEMA}]
    messages = [{"role": "developer", "content": developer}]
    errors: Counter[str] = Counter()
    transcript: list[dict] = []
    stop = "max_turns"

    obs = R.render_env(board, style)
    messages.append({"role": "user", "content": obs})
    harmony_msgs.append({"role": "user", "content": obs})
    if keep_transcript:
        transcript.append({"role": "user", "content": obs})

    for turn in range(max_turns):
        reply = client.call(messages, tools, seed=seed, harmony=harmony_msgs)
        if "error" in reply:
            # A 4xx (e.g. "maximum context length" once an unbounded episode
            # outgrows max-model-len) is the episode legitimately running out
            # of room, not a server malfunction -- keep it out of
            # "model_error" so that count stays a signal for genuine problems.
            if 400 <= (reply.get("http_status") or 0) < 500:
                errors["context_exceeded"] += 1
                stop = "context_exceeded"
            elif reply.get("malformed_tool_call"):
                errors["malformed_tool_call"] += 1
                stop = "malformed_tool_call"
            else:
                errors["model_error"] += 1
                stop = "model_error"
            break
        choice = (reply.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        calls = msg.get("tool_calls") or []
        raw = {
            "turn": turn,
            "arguments": (((calls[0].get("function") or {}).get("arguments"))
                          if calls else None),
            "tool_name": (((calls[0].get("function") or {}).get("name"))
                          if calls else None),
            "content": msg.get("content"),
            "reasoning": msg.get("reasoning_content"),
            "finish_reason": choice.get("finish_reason"),
            "usage": (reply.get("usage") or {}).get("completion_tokens"),
        }
        output.append(raw)
        if keep_transcript:
            transcript.append({"role": "assistant", "content": msg.get("content"),
                               "tool_calls": calls,
                               "reasoning": msg.get("reasoning_content")})
        if not calls:
            # No call is the stop decision: the model answered instead.
            stop = "final"
            if msg.get("content"):
                harmony_msgs.append({"role": "assistant", "channel": "final",
                                     "content": msg["content"], "supervised": True})
            break
        arguments = ((calls[0].get("function") or {}).get("arguments")) or "{}"
        # Carry the analysis back into the history.  A reasoning-trained board
        # model writes analysis before every tool call, so a history whose
        # assistant turns have none is off its training distribution: it reasons
        # on turn 1, sees a transcript in which it never did, and stops -- the
        # reasoning arm would be switched off after its first move.  Feeding it
        # back keeps it on.  Harmless for the no-reasoning arm, whose
        # reasoning_content is empty anyway.
        prior = {"role": "assistant", "tool_calls": calls}
        if msg.get("reasoning_content"):
            prior["reasoning_content"] = msg["reasoning_content"]
        messages.append(prior)
        if msg.get("reasoning_content"):
            harmony_msgs.append({
                "role": "assistant", "channel": "analysis",
                "content": msg["reasoning_content"], "supervised": True,
            })
        harmony_msgs.append({
            "role": "assistant", "channel": "commentary",
            "recipient": f"functions.{H.TOOL_NAME}", "content_type": "json",
            "content": arguments, "supervised": True,
        })
        try:
            actions = parse_act_json(arguments)
        except ActError as exc:
            errors["unparsed_call"] += 1
            rej = f"rejected: {exc}"
            raw["rejected"] = rej
            messages.append({"role": "tool", "name": H.TOOL_NAME, "content": rej})
            harmony_msgs.append({"role": "tool", "name": f"functions.{H.TOOL_NAME}",
                                 "channel": "commentary", "content": rej})
            if keep_transcript:
                transcript.append({"role": "tool", "content": rej})
            if errors["unparsed_call"] >= 3:
                stop = "unparsed_call"
                break
            continue
        if any(isinstance(a, Terminate) for a in actions):
            # `done` claims a route and the episode carries on; only `terminate`
            # stops -- and a model that stops by answering instead of calling is
            # handled by the `not calls` branch above.
            stop = "terminate"
            break
        try:
            board.apply(actions)
        except NoMenu as exc:
            errors["no_menu"] += 1
            rej = f"rejected: {exc}"
            raw["rejected"] = rej
            messages.append({"role": "tool", "name": H.TOOL_NAME, "content": rej})
            harmony_msgs.append({"role": "tool", "name": f"functions.{H.TOOL_NAME}",
                                 "channel": "commentary", "content": rej})
            if keep_transcript:
                transcript.append({"role": "tool", "content": rej})
            continue
        except BoardError as exc:
            errors["illegal_action"] += 1
            rej = f"rejected: {exc}"
            raw["rejected"] = rej
            errors[f"illegal:{str(exc).split(' ', 1)[-1][:40]}"] += 1
            messages.append({"role": "tool", "name": H.TOOL_NAME, "content": rej})
            harmony_msgs.append({"role": "tool", "name": f"functions.{H.TOOL_NAME}",
                                 "channel": "commentary", "content": rej})
            if keep_transcript:
                transcript.append({"role": "tool", "content": rej})
            if errors["illegal_action"] >= args_illegal_cap():
                stop = "illegal_action"
                break
            continue
        if t_first_route is None and board.routes:
            # Time to the FIRST route, not to the end of the episode: an agent
            # that finds one route quickly and then keeps looking is a different
            # proposition from one that takes as long to find its first.
            t_first_route = time.time() - t_start
            # Call count at the same moment, for post-hoc binning against a
            # budget axis: one run to a high ceiling lets success at any budget
            # B <= ceiling be read off afterwards instead of re-running per B.
            first_route_calls = board.budget_used
        obs = R.render_env(board, style)
        messages.append({"role": "tool", "name": H.TOOL_NAME, "content": obs})
        harmony_msgs.append({"role": "tool", "name": f"functions.{H.TOOL_NAME}",
                             "channel": "commentary", "content": obs})
        if keep_transcript:
            transcript.append({"role": "tool", "content": obs})
        if board.budget_used >= budget:
            stop = "budget"
            break
        if not board.open_mols() and not board.solved():
            # Nothing to act on and no route: every branch is either committed
            # and waiting on a piece that cannot be reached, or dead.  Spinning
            # here burns the illegal-action budget on a state with no legal move.
            stop = "stuck"
            break

    # `auto_routes` is off, so the board registers a route only when the model
    # CLAIMS one -- stopping is the agent's decision to make.  But a model that
    # closed the root and then stopped has still BUILT a route; it just never
    # said the word.  Without this, `solved` and `routes` disagree, and a
    # prompted baseline that does not know the claim verb scores zero routes on
    # chemistry it actually did, while a model trained on the protocol scores every one of its own.  That
    # gap would be protocol fluency reported as chemistry.
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
        priced = [board.mols[m].price_ln for m in leaves
                  if board.mols[m].price_ln is not None]
        import math
        routes.append({
            "label": route.label, "n_steps": len(rxns),
            "weakest": min(vals) if vals else None,
            "n_leaves": len(leaves),
            "n_unpriced": len(leaves) - len(priced),
            "cost_usd": round(sum(math.exp(v) for v in priced), 2) if priced else None,
            "steps": [[board.mols[board.rxns[r].parent].smiles,
                       [board.mols[p].smiles for p in board.rxns[r].pieces]]
                      for r in rxns],
        })

    # Every step the search APPLIED, with the material a reaction-family identity is computed
    # from. The transcript alone cannot give this: the board prints SMILES and scores, never a
    # reaction name, so `family_key` is computable on the training episodes (which carry
    # `evidence`) and not on a rollout. Without it the diversity axis -- distinct reaction
    # families found per call, against the number the menus OFFERED -- can be measured on the
    # corpus and not on the checkpoint, which is the wrong way round.
    applied_steps = []
    for rid in board.rxn_order:
        rxn = board.rxns.get(rid)
        if rxn is None:
            continue
        parent = board.mols.get(rxn.parent)
        cand = next((c for c in (parent.menu or []) if c.idx == rxn.cand), None) \
            if parent is not None else None
        row = {"rid": rid, "mid": rxn.parent, "cand": rxn.cand, "status": rxn.status}
        if parent is not None and cand is not None:
            try:
                fx = board.evidence.for_step(parent.smiles, cand.reactants)
                nm = fx.get("named") or {}
                bond = fx.get("bond") or {}
                row["named_tier"] = nm.get("tier")
                row["named"] = nm.get("names") or []
                row["formed"] = sorted({(b or {}).get("atoms") for b in
                                        (bond.get("formed") or []) if (b or {}).get("atoms")})
                row["broken"] = sorted({(b or {}).get("atoms") for b in
                                        (bond.get("broken") or []) if (b or {}).get("atoms")})
            except Exception as e:                                     # noqa: BLE001
                row["family_error"] = f"{type(e).__name__}: {e}"[:120]
        applied_steps.append(row)
    # The denominator: the families the MENUS put on screen, taken or not. Without it
    # "families found" rewards a deeper search rather than a more varied one.
    offered_steps = []
    for mid, m in board.mols.items():
        for c in (m.menu or []):
            try:
                fx = board.evidence.for_step(m.smiles, c.reactants)
                nm = fx.get("named") or {}
                bond = fx.get("bond") or {}
                offered_steps.append({
                    "mid": mid, "cand": c.idx, "named_tier": nm.get("tier"),
                    "named": nm.get("names") or [],
                    "formed": sorted({(b or {}).get("atoms") for b in
                                      (bond.get("formed") or []) if (b or {}).get("atoms")}),
                    "broken": sorted({(b or {}).get("atoms") for b in
                                      (bond.get("broken") or []) if (b or {}).get("atoms")})})
            except Exception:                                          # noqa: BLE001
                continue

    out = {
        "target": target, "rollout": rollout, "stock": world.stock_name,
        "applied_steps": applied_steps, "offered_steps": offered_steps,
        "solved": board.solved(), "stop": stop,
        "turns": turn + 1, "calls": board.budget_used,
        "wall_s": round(time.time() - t_start, 2),
        "first_route_s": (round(t_first_route, 2)
                          if t_first_route is not None else None),
        "first_route_calls": first_route_calls,
        "n_open_left": len(board.open_mols()),
        "n_dead": len(board.dead_mols()),
        "routes": routes, "routes_implicit": implicit,
        "errors": dict(errors),
        "output": output,
    }
    if keep_messages:
        out["messages"] = [{"role": "developer", "content": developer}] + [
            m for m in messages if m["role"] != "developer"]
        out["harmony_messages"] = harmony_msgs
        out["tools"] = tools
        out["reasoning_effort"] = client.reasoning_effort
        if wire is not None:
            # The episode may have died mid-call, so a render failure is recorded
            # rather than raised: the run's numbers do not depend on it.
            try:
                rendered = wire(out)
            except Exception as exc:
                out["wire_text_error"] = f"{type(exc).__name__}: {exc}"
            else:
                if rendered:
                    out.update(rendered)
    if keep_transcript:
        out["transcript"] = transcript
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--model-url", required=True,
                    help="comma-separated OpenAI-compatible /v1 bases")
    ap.add_argument("--model", required=True)
    ap.add_argument("--menu-url", required=True,
                    help="comma-separated /predict URLs of the single-step server")
    ap.add_argument("--stock", default="emols",
                    help="the evaluation stock is `emols` (the eMolecules "
                         "catalogue, full InChIKey). `train_paroutes` is the "
                         "training-leaf set: it closes routes the benchmark stock "
                         "would not, so a run against it is not on this benchmark. "
                         "Every dump records which one it used.")
    ap.add_argument("--menu-cache", default=None,
                    help="NOT for evaluation. Seeds the menu cache from a recorded "
                         "draw_cache, which makes the run partly replayed rather "
                         "than live -- the single-step server is the thing being "
                         "evaluated alongside the agent, so it has to be called. "
                         "Kept only for debugging the harness cheaply; scale "
                         "replicas instead.")
    ap.add_argument("--rt", default=None,
                    help="live:URL to score round-trip on the menu; default is the "
                         "recorded cache only")
    ap.add_argument("--illegal-cap", type=int, default=30,
                    help="refused actions tolerated before the episode is "
                         "abandoned. 1 = NO-RETRY: the episode ends on the first "
                         "refused action")
    ap.add_argument("--price-live", action="store_true",
                    help="price a buyable molecule the cache does not carry with "
                         "MolPrice, instead of rendering it with no `$`. Off by "
                         "default: it changes the cost axis, so a run with it is not "
                         "comparable with one without. CPU only.")
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--menu-show", type=int, default=10)
    ap.add_argument("--signals", default="q,p,rt")
    ap.add_argument("--deciding", default="p")
    ap.add_argument("--handover", default="flow",
                    choices=["tree", "steps", "nest", "sexp", "flow", "table"],
                    help="how the final message renders the answer. Has to match "
                         "the corpus the checkpoint was trained on -- "
                         "check_wire_match.py compares the preamble, but this "
                         "shows up only in the last message, so it is on the "
                         "operator to keep them the same")
    ap.add_argument("--cutoff", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--reasoning", default="medium",
                    help="the `reasoning_effort` sent with every request, which is what the "
                         "server turns into the `Reasoning: <x>` line of the system message. IT "
                         "MUST MATCH THE VALUE THE CORPUS WAS STAMPED WITH -- see "
                         "traj_route_reasoning_routeloop.py --reasoning-effort. A reasoning "
                         "checkpoint served below the effort it was trained at does not open "
                         "the analysis channel, and the run measures the non-reasoning model "
                         "instead. The training corpus is stamped `medium`, so this defaults "
                         "to medium")
    ap.add_argument("--reasoning-form", action="store_true",
                    help="developer message describes the schema-form reasoning "
                         "block ('How to think, before each call'). Required for "
                         "any checkpoint whose training corpus was rendered with "
                         "board/harmony.py's analysis_form=True (schema-reasoning "
                         "checkpoints) -- served without it, the model "
                         "gets a developer message it never trained on and the "
                         "mismatch is silent (see check_wire_match.py)")
    ap.add_argument("--raw-completions", action="store_true",
                    help="build the prompt with verify_harmony.build and call "
                         "/v1/completions, instead of letting vllm's chat template "
                         "re-render the conversation on /v1/chat/completions. The "
                         "chat rendering is NOT the one these corpora were written "
                         "with -- it stamps a `Current date:` line into the system "
                         "message and renders the tool schema differently -- and a "
                         "checkpoint served that prefix can return tokens the tool "
                         "parser cannot read, which reports as unsolved targets "
                         "rather than as an error.")
    ap.add_argument("--developer-file", default=None,
                    help="use this file's content VERBATIM as the developer "
                         "message instead of regenerating it from "
                         "board/harmony.py. Needed when that module has moved on "
                         "since the checkpoint's corpus was rendered -- pull the "
                         "text from any row's `wire_developer`/`messages[0]` so "
                         "the eval serves exactly what was trained on")
    ap.add_argument("--rollouts", type=int, default=1,
                    help="independent rollouts per target. pass@k in the "
                         "retrosynthesis literature is over k ROLLOUTS, not over "
                         "the routes one rollout returns, so k>1 needs sampling: "
                         "--temperature 0 makes every rollout identical and pass@k "
                         "collapses to pass@1 by construction")
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed; rollout i of a target is seeded seed+i so the "
                         "run is reproducible")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("RP_WORKERS", "8")))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--transcripts", action="store_true")
    ap.add_argument("--no-raw-text", action="store_true",
                    help="keep the conversation but skip the wire_text/raw_text "
                         "renderings, which is most of the dump's bytes and is "
                         "what makes an eval episode diffable against a training "
                         "one")
    ap.add_argument("--no-messages", action="store_true",
                    help="omit the full conversation from the dump. It is on by "
                         "default: `output[]` alone records what the model said "
                         "but not what it was shown, so a rejection cannot be "
                         "traced to the board that produced it. Cost is real -- "
                         "the boards are most of the bytes -- and `raw_text` is "
                         "added by verify_harmony.py --augment afterwards")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    _ILLEGAL_CAP[0] = args.illegal_cap
    developer_override = (Path(args.developer_file).read_text()
                          if args.developer_file else None)

    targets = []
    with open(args.targets) as fh:
        for line in fh:
            row = json.loads(line)
            targets.append(row["target"] if isinstance(row, dict) else row)
    if args.limit:
        targets = targets[: args.limit]

    scores = load_node_scores(want=("rt", "price", "plaus"))
    # Training/serving parity, printed before the run starts. Both halves are silent
    # when wrong: served below the effort it was trained at the model emits no analysis and the
    # run measures the non-reasoning model; served without --reasoning-form the developer block
    # never describes the schema and the model imitates a form it was not told.
    print(f"# serving reasoning_effort={args.reasoning}  reasoning_form="
          f"{'on' if args.reasoning_form else 'OFF'}", file=sys.stderr)
    if args.reasoning == "low":
        print("# !! effort=low: a reasoning checkpoint served at low does not open the "
              "analysis channel. Pass --reasoning medium to match the training corpus.",
              file=sys.stderr)
    if not args.reasoning_form:
        print("# !! --reasoning-form is OFF: required for any checkpoint whose corpus was "
              "rendered with analysis_form=True (every schema-reasoning run).", file=sys.stderr)
    stock = C.Stock(args.stock)
    print(f"# stock {args.stock}: {len(stock):,} keys ({stock.mode})", file=sys.stderr)
    print(f"# caches: plaus {len(scores.plaus):,} · rt {len(scores.rt):,} "
          f"· price {len(scores.price):,}", file=sys.stderr)

    rt_url = args.rt.split("live:", 1)[1] if (args.rt or "").startswith("live:") else None
    seed_menus = {}
    if args.menu_cache:
        import json as _json
        raw = _json.loads(Path(args.menu_cache).read_text())
        seed_menus = {k: [(list(r), float(q)) for r, q in v] for k, v in raw.items()}
        print(f"# menu cache seeded with {len(seed_menus):,} molecules",
              file=sys.stderr)
    world = LiveWorld(
        [u.strip() for u in args.menu_url.split(",") if u.strip()],
        stock=stock, stock_name=args.stock, prices=scores.price, rt_cache=scores.rt,
        plaus_cache=scores.plaus, top_k=args.menu_show, rt_url=rt_url,
        price_live=args.price_live,
    )
    world.menus.update(seed_menus)
    Client = RawCompletionClient if args.raw_completions else ChatClient
    client = Client([u.strip() for u in args.model_url.split(",") if u.strip()],
                        args.model, temperature=args.temperature,
                        max_tokens=args.max_tokens, reasoning_effort=args.reasoning)
    style = R.RenderStyle(
        menu_show=args.menu_show, menu_cutoff=args.cutoff,
        signal_order=tuple(args.signals.split(",")),
        cutoff_signal=args.deciding, route_signal=args.deciding,
        handover=args.handover,
    )
    wire = None if (args.no_messages or args.no_raw_text) else WireText(style)
    if wire is not None and not wire.ok:
        print(f"# wire_text unavailable ({wire.reason}); run verify_harmony.py "
              f"{args.out} --augment ... to add it", file=sys.stderr)
        wire = None
    elif wire is not None and wire.enc is None:
        print(f"# wire_text unavailable ({wire.reason}); raw_text only",
              file=sys.stderr)

    done = 0
    tally: Counter[str] = Counter()
    lock = threading.Lock()
    out_f = open(args.out, "w")

    jobs = [(t, i) for t in targets for i in range(args.rollouts)]
    if args.rollouts > 1 and args.temperature == 0.0:
        print("!! --rollouts > 1 with temperature 0: every rollout will be "
              "identical and pass@k cannot differ from pass@1", file=sys.stderr)

    def work(job):
        target, roll = job
        try:
            return run_target(target, world, client, style, budget=args.budget,
                              max_depth=args.max_depth, max_turns=args.max_turns,
                              deciding=args.deciding,
                              keep_transcript=args.transcripts,
                              keep_messages=not args.no_messages, wire=wire,
                              rollout=roll, seed=args.seed + roll,
                              reasoning_form=args.reasoning_form,
                              developer_override=developer_override)
        except Exception as exc:                       # noqa: BLE001
            return {"target": target, "rollout": roll, "solved": False,
                    "stop": "crash", "error": f"{type(exc).__name__}: {exc}",
                    "routes": [], "errors": {}, "turns": 0, "calls": 0}

    from concurrent.futures import as_completed

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, j) for j in jobs]
        for fut in as_completed(futures):
            row = fut.result()
            with lock:
                done += 1
                tally["solved"] += bool(row["solved"])
                tally[f"stop_{row['stop']}"] += 1
                tally["calls"] += row.get("calls", 0)
                for k, v in (row.get("errors") or {}).items():
                    tally[f"err_{k}"] += v
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                if done % 25 == 0:
                    print(f"# {done}/{len(jobs)} solved={tally['solved']} "
                          f"({tally['solved'] / done:.1%}) "
                          f"mean_calls={tally['calls'] / done:.1f}",
                          file=sys.stderr, flush=True)
    out_f.close()
    # The scaled interval leaves a tail unflushed; without this the last window of live
    # round-trip work is lost.
    world._flush_rt()
    if world._rt_new:
        print(f"# rt overlay: {len(world._rt_new):,} newly scored reactions -> "
              f"{world._rt_overlay}", file=sys.stderr)

    n = max(done, 1)
    print(f"\n=== {len(targets)} targets x {args.rollouts} rollouts "
          f"(T={args.temperature}) · stock {args.stock} · budget {args.budget} ===")
    print(f"solve rate      {tally['solved']}/{done} ({tally['solved'] / n:.1%})")
    print(f"mean calls      {tally['calls'] / n:.1f}")
    print("stop reasons    " + " · ".join(
        f"{k[5:]} {v}" for k, v in sorted(tally.items()) if k.startswith("stop_")))
    errs = {k[4:]: v for k, v in tally.items() if k.startswith("err_")}
    print(f"rejections      {errs or 'none'}")
    print(f"world           {dict(world.stats)}")
    print(f"model           {dict(client.stats)}")
    print(f"# wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
