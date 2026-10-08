from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp import FastMCP


def add_transport_args(parser: argparse.ArgumentParser, default_port: int) -> None:
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "http"),
        default="stdio",
        help="Transport: stdio (default), sse, or http (streamable-http).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=default_port)


def run_with_args(mcp: "FastMCP", args: argparse.Namespace) -> None:
    if args.transport == "stdio":
        mcp.run()
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    elif args.transport == "http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        raise ValueError(f"unknown transport: {args.transport}")
