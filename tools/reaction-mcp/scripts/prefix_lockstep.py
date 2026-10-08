"""Arrival check: does this checkout rebuild the prefix and the board the checkpoint was trained on?

Run this BEFORE the first eval on a new checkpoint. A rendering that has drifted does
not raise -- it returns a board the model has never seen, the model emits nothing the
parser recognises, and the run reports a clean zero rather than an error.

GOLDEN is a JSON dump of the rendered prefix and board from the training corpus; DEV is the
developer message the eval serves:

    cd tools/reaction-mcp/scripts; . ../../../config/env.sh
    BOARD_COST_NORM=1 BOARD_ROUTES_FULL=1 BOARD_ROUTE_AXES=1 \
      python prefix_lockstep.py GOLDEN.json $RP_PROTOCOL/developer/dev_retroplanner.txt

Exits non-zero on any mismatch.
"""
from __future__ import annotations

import difflib
import json
import os
import pathlib
import re
import sys

GOLDEN, DEV = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# config/paths.py, loaded by file under its own name rather than put on sys.path as
# `paths`, where any other module of that name would shadow it.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "rp_paths", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "config", "paths.py"))
RP = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(RP)
sys.path.insert(0, RP.BOARD)                                       # the board/ package

import board.harmony as H                                          # noqa: E402
import board.render as R                                           # noqa: E402
import verify_harmony as V                                         # noqa: E402
from openai_harmony import (HarmonyEncodingName, Role,             # noqa: E402
                            load_harmony_encoding)

ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
gold = json.load(open(GOLDEN))
bad = 0


def report(what: str, want: str, got: str) -> None:
    global bad
    if want == got:
        print(f"  ok    {what:26s} {len(want):6d} chars")
        return
    bad += 1
    print(f"  FAIL  {what:26s} want {len(want)} chars, got {len(got)}")
    d = [ln for ln in difflib.unified_diff(want.splitlines(), got.splitlines(),
                                           "trained on", "this checkout",
                                           lineterm="", n=0)]
    print(f"        {sum(1 for ln in d if ln[:1] in '+-' and ln[:3] not in ('+++', '---'))}"
          " changed lines; first few:")
    for ln in d[:12]:
        print("        ", ln[:130])


# 1. the harmony prefix -- developer body from the shipped file, tool namespace
#    generated from board.harmony.ACT_SCHEMA, system from the effort stamp.
tools = [{"name": H.TOOL_NAME, "description": H.TOOL_DESC, "parameters": H.ACT_SCHEMA}]
convo = V.build({"messages": [{"role": "developer",
                               "content": pathlib.Path(DEV).read_text()}],
                 "tools": tools, "harmony_messages": [], "reasoning_effort": "medium"})
text = ENC.decode(ENC.render_conversation_for_completion(convo, Role.ASSISTANT))


def section(tag: str) -> str:
    m = re.search(rf"<\|start\|>{tag}<\|message\|>(.*?)<\|end\|>", text, re.S)
    return m.group(1) if m else ""


print(f"prefix, against {gold['arm']}")
report("developer", gold["wire_developer"], section("developer"))
report("system", gold["wire_system"], section("system"))

# 2. the render flags.  These reach render.py through the ENVIRONMENT and nothing
#    else -- the eval scripts never set them -- so a launcher that forgets one
#    serves the right glossary over the wrong board, which is the worst of the
#    failure modes because the prompt check above still passes.
print(f"\nrender flags, against {len(gold['observations'])} boards from a trained episode")
OBS = "\n".join(gold["observations"])
for name, live, shown in (
        ("BOARD_ROUTE_AXES", R.ROUTE_AXES(), bool(re.search(r"\d+/\d+ pass", OBS))),
        ("BOARD_ROUTE_FRONT", R.ROUTE_FRONT(), "· front " in OBS),
        ("BOARD_COST_NORM", R.COST_NORM(), not re.search(r"\$\s?\d{2,}", OBS)),
):
    ok = live == shown
    bad += not ok
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:20s} this checkout {str(live):5s} "
          f"· the trained board {'shows' if shown else 'does not show'} it")
# ROUTES_FULL only shows on a board that claims a second route, which this one
# episode may not reach, so it is reported rather than asserted.
print(f"  note  BOARD_ROUTES_FULL    this checkout {R.ROUTES_FULL()} "
      "(not checkable from one episode)")

print("\n" + ("MISMATCH -- do not run the eval" if bad else "lockstep ok"))
sys.exit(1 if bad else 0)
