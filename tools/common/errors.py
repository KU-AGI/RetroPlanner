from __future__ import annotations

from enum import Enum
from typing import Any


class ToolErrorType(str, Enum):
    INVALID_INPUT = "invalid_input"
    PARSE_ERROR = "parse_error"
    NOT_FOUND = "not_found"
    UPSTREAM_ERROR = "upstream_error"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"
    INTERNAL_ERROR = "internal_error"


def error_envelope(message: str, error_type: ToolErrorType | str, **extra: Any) -> dict:
    """Standard recoverable-error payload returned from a tool.

    Tools should return this when the failure is part of expected operating
    behavior (e.g. malformed SMILES, compound not found). Unexpected exceptions
    should propagate so FastMCP wraps them into ToolError.
    """
    t = error_type.value if isinstance(error_type, ToolErrorType) else str(error_type)
    payload: dict[str, Any] = {"error": message, "error_type": t}
    payload.update(extra)
    return payload
