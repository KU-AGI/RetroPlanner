from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

from common import add_transport_args, run_with_args  # noqa: E402

from .server import mcp, warmup  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(prog="reaction-mcp")
    add_transport_args(parser, default_port=8004)
    args = parser.parse_args()
    # Pre-load in-process models in the background so the server is responsive
    # immediately while rxnmapper and the template corpus warm (REACTION_WARMUP=0 to disable).
    import threading

    threading.Thread(target=warmup, name="reaction-warmup", daemon=True).start()
    run_with_args(mcp, args)


if __name__ == "__main__":
    main()
