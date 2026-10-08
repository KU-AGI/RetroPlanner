#!/usr/bin/env python
"""Render board episodes with openai_harmony itself, and check they survive it.

The builder emits structured messages, never Harmony token text (the token layer
moves between library versions).  This is the one place that touches it:

  * renders each episode through openai_harmony and reports its token length,
  * asserts every tool call parses back to the same actions,
  * asserts no message body contains a Harmony special token,
  * optionally writes the rendered text out for eyeballing.

Runs in an env that HAS openai_harmony (not the rdkit builder env):

  $CONDA_ROOT/vllm14/bin/python scripts/verify_harmony.py \
      <episodes>.jsonl --dump <episode>.harmony.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SD = Path(__file__).resolve().parent
sys.path.insert(0, str(SD))
# config/paths.py, loaded by file under its own name rather than put on sys.path as
# `paths`, where any other module of that name would shadow it.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "rp_paths", str(Path(__file__).resolve().parents[3] / "config" / "paths.py"))
RP = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(RP)
sys.path.insert(0, RP.BOARD)                          # the board/ package lives with the harness

from board.parse import ActError, parse_act_json     # noqa: E402

from openai_harmony import (Author, Conversation, DeveloperContent,   # noqa: E402
                            HarmonyEncodingName, Message, ReasoningEffort, Role,
                            SystemContent, ToolDescription, load_harmony_encoding)

EFFORT = {"low": ReasoningEffort.LOW, "medium": ReasoningEffort.MEDIUM,
          "high": ReasoningEffort.HIGH}
SPECIAL = ("<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|call|>",
           "<|return|>", "<|constrain|>")
SYS_RE = re.compile(r"<\|start\|>system<\|message\|>(.*?)<\|end\|>", re.DOTALL)
DEV_RE = re.compile(r"<\|start\|>developer<\|message\|>(.*?)<\|end\|>", re.DOTALL)


_EFFORT_WARNED = set()


def _effort(row: dict) -> str:
    """The row's reasoning effort -- and a complaint if it does not state one.

    A silent default would render a row that stated nothing as `Reasoning: low`, looking
    exactly like a row that meant it. The effort goes into the system message the student is
    trained under and the eval must be served at the same value, so guessing it wrong is a
    distribution shift with no symptom. Missing is reported once per run and still falls back, because refusing to render a corpus
    over a metadata field would be worse than saying so.
    """
    v = row.get("reasoning_effort")
    if v in EFFORT:
        return v
    key = repr(v)
    if key not in _EFFORT_WARNED:
        _EFFORT_WARNED.add(key)
        print(f"!! rows carry reasoning_effort={key}, which is not one of {sorted(EFFORT)}; "
              f"rendering them as 'low'. The system message the student trains under says the "
              f"effort, and the eval has to be served the same value -- stamp it with "
              f"`traj_route_reasoning_*.py --reasoning-effort`.", file=sys.stderr)
    return "low"


def build(row: dict) -> Conversation:
    """The vLLM-identical Harmony rendering of one row.

    Reads `harmony_messages` (Harmony's own field names) and takes the developer
    instructions from `messages[0]`, which is where apply_chat_template wants
    them -- so one row feeds both paths and they cannot drift apart.
    """
    sysc = (SystemContent.new()
            .with_reasoning_effort(EFFORT[_effort(row)]))
    instructions = next((m["content"] for m in row["messages"]
                         if m["role"] in ("developer", "system")), "")
    devc = (DeveloperContent.new()
            .with_instructions(instructions)
            .with_function_tools([
                ToolDescription.new(t["name"], t["description"], parameters=t["parameters"])
                for t in row.get("tools", [])
            ]))
    msgs = [Message.from_role_and_content(Role.SYSTEM, sysc),
            Message.from_role_and_content(Role.DEVELOPER, devc)]
    for m in row["harmony_messages"]:
        if m["role"] == "user":
            msgs.append(Message.from_role_and_content(Role.USER, m["content"]))
        elif m["role"] == "tool":
            # vLLM's own path (entrypoints/openai/parser/harmony_utils.py) sets
            # channel=commentary AND recipient=assistant, which is what puts the
            # " to=assistant" in the header the model actually sees at inference.
            msg = (Message.from_author_and_content(Author.new(Role.TOOL, m["name"]),
                                                   m["content"])
                   .with_channel(m.get("channel", "commentary"))
                   .with_recipient("assistant"))
            msgs.append(msg)
        elif m["role"] == "assistant":
            msg = Message.from_role_and_content(Role.ASSISTANT, m["content"])
            if m.get("channel"):
                msg = msg.with_channel(m["channel"])
            if m.get("recipient"):
                msg = msg.with_recipient(m["recipient"])
            if m.get("content_type"):
                msg = msg.with_content_type(m["content_type"])
            msgs.append(msg)
        else:
            raise ValueError(f"unknown role {m['role']!r}")
    return Conversation.from_messages(msgs)


def first_text_of(enc, toks) -> str:
    return enc.decode(toks)


def _raise(msg):
    raise AssertionError(msg)


def _assistant_spans(enc, convo, row) -> list[dict]:
    """Token spans to compute loss on: one per assistant message, plus its flag.

    Computed by rendering growing prefixes, so the boundaries are the encoder's
    and not a re-tokenisation of decoded text.  An unsupervised span (a rollout
    mistake or a probe) is reported with supervised=False so the trainer can mask
    it instead of teaching the mistake.
    """
    # build() emits system, developer, then one conversation message per row
    # message -- so conversation index k >= 2 is row["messages"][k - 2].  Getting
    # this offset wrong silently marks every span supervised, which is the one
    # failure mode that cannot be seen in the rendered text.
    msgs = convo.messages
    assert len(msgs) == len(row["harmony_messages"]) + 2, \
        "message count drifted from build()"
    prev = len(enc.render_conversation(Conversation.from_messages(msgs[:2])))
    spans = []
    for k in range(2, len(msgs)):
        n = len(enc.render_conversation(Conversation.from_messages(msgs[: k + 1])))
        src = row["harmony_messages"][k - 2]
        if src["role"] == "assistant":
            spans.append({"start": prev, "end": n,
                          "supervised": bool(src.get("supervised", True)),
                          "channel": src.get("channel")})
        prev = n
    return spans


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--dump", default=None, help="write the first episode's tokens as text")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--compare-hf", default=None, metavar="CHAT_TEMPLATE_JINJA",
                    help="also render hf_messages through openai/gpt-oss's own "
                         "chat_template.jinja and report where the two disagree")
    ap.add_argument("--augment", default=None,
                    help="write each input row PLUS wire_system / wire_developer / "
                         "wire_text / spans. The preamble openai_harmony renders "
                         "is what vLLM feeds the model, and it is NOT what the HF "
                         "chat template renders (the tool namespace differs), so "
                         "this is the file to train on")
    ap.add_argument("--render-out", default=None,
                    help="write {target, text, tokens, spans} per episode: the wire "
                         "text plus the assistant token spans to compute loss on")
    args = ap.parse_args()

    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    tmpl = None
    if args.compare_hf:
        from jinja2 import Environment
        env = Environment(trim_blocks=False, lstrip_blocks=False)
        env.globals["raise_exception"] = _raise
        env.globals["strftime_now"] = lambda f: "2026-08-23"
        tmpl = env.from_string(Path(args.compare_hf).read_text())
    lens, calls, bad = [], 0, 0
    first_text = None
    hf_delta = []
    out_f = open(args.render_out, "w") if args.render_out else None
    aug_f = open(args.augment, "w") if args.augment else None
    for i, line in enumerate(open(args.jsonl)):
        if args.limit and i >= args.limit:
            break
        row = json.loads(line)
        for m in row["harmony_messages"]:
            for tok in SPECIAL:
                if tok in (m.get("content") or ""):
                    print(f"!! special token {tok} inside a message body", file=sys.stderr)
                    bad += 1
            if m["role"] == "assistant" and m.get("recipient"):
                calls += 1
                try:
                    parse_act_json(m["content"])
                except ActError as exc:
                    print(f"!! tool call does not parse: {exc}", file=sys.stderr)
                    bad += 1
        convo = build(row)
        toks = enc.render_conversation(convo)
        lens.append(len(toks))
        if first_text is None:
            first_text = enc.decode(toks)

        if tmpl is not None:
            hf = row.get("raw_text") or tmpl.render(
                messages=row["messages"], add_generation_prompt=False,
                reasoning_effort=_effort(row),
                tools=[{"type": "function", "function": t}
                       for t in row.get("tools", [])],
            )
            # The number that matters is not the length, it is whether the board
            # reaches the model as written.  harmony passes it through; the
            # template runs it through an html-safe |tojson.
            board = next((m["content"] for m in row["harmony_messages"]
                          if m["role"] == "tool"), None)
            hf_delta.append((
                board is not None and board in hf,
                board is not None and board in first_text_of(enc, toks),
                len(enc.encode(hf, allowed_special="all")), len(toks),
            ))

        if aug_f is not None:
            text = enc.decode(toks)
            sysm = SYS_RE.search(text)
            devm = DEV_RE.search(text)
            if not sysm or not devm:
                print("!! could not split the preamble", file=sys.stderr)
                bad += 1
            else:
                row["wire_system"] = sysm.group(1)
                row["wire_developer"] = devm.group(1)
                row["wire_text"] = text
                row["spans"] = _assistant_spans(enc, convo, row)
                aug_f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if out_f is not None:
            out_f.write(json.dumps({
                "target": row["target"],
                "text": enc.decode(toks),
                "tokens": len(toks),
                "spans": _assistant_spans(enc, convo, row),
            }, ensure_ascii=False) + "\n")

    if out_f is not None:
        out_f.close()
        print(f"# wrote {args.render_out}")
    if aug_f is not None:
        aug_f.close()
        print(f"# wrote {args.augment}")
    if hf_delta:
        hf_raw = sum(1 for a, _, _, _ in hf_delta if a)
        hy_raw = sum(1 for _, b, _, _ in hf_delta if b)
        n = len(hf_delta)
        print(f"# board text reaches the model verbatim: harmony {hy_raw}/{n} · "
              f"hf chat_template {hf_raw}/{n}")
        print(f"# (the template renders a tool result as content|tojson, html-safe: "
              f"'<c0 ...>' becomes '\\u003cc0 ...' and every newline an escape -- "
              f"more tokens on a board, and NOT what vLLM feeds at inference)")
    if args.dump and first_text is not None:
        Path(args.dump).write_text(first_text)
        print(f"# wrote {args.dump}")
    lens.sort()
    if lens:
        print(f"# {len(lens)} episodes · tokens min {lens[0]:,} "
              f"median {lens[len(lens)//2]:,} max {lens[-1]:,}")
    print(f"# {calls} tool calls · {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
