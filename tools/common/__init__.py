"""Shared helpers across chemistry MCP servers.

Importable when ``tools/`` is on ``sys.path``. Each server's ``__main__.py``
adjusts ``sys.path`` to make this package available without an extra install.
"""

from .errors import error_envelope, ToolErrorType
from .formatting import (
    ERROR_TEMPLATE,
    TOOL_TEMPLATES,
    format_output,
    parse_text_output,
    register_template,
    template_placeholders,
    template_to_regex,
)
from .transport import add_transport_args, run_with_args

__all__ = [
    "error_envelope",
    "ToolErrorType",
    "add_transport_args",
    "run_with_args",
    "format_output",
    "register_template",
    "template_to_regex",
    "template_placeholders",
    "parse_text_output",
    "TOOL_TEMPLATES",
    "ERROR_TEMPLATE",
]
