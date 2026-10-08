"""Dual-mode tool output formatting.

Each MCP tool returns its result either as a JSON-like dict (the default,
preserved for backward compatibility) or as a wrapped *text* string driven by
a per-tool template. The template uses ``str.format`` style ``{placeholder}``
slots that map 1:1 to keys in the tool's result dict.

Why a template:
- Verifier / reward functions can introspect ``TOOL_TEMPLATES`` to
  discover, for each tool, the exact text shape its output will take. From
  that shape we derive a regex (``template_to_regex``) whose named groups
  recover the original values from the rendered text. This lets a reward
  function score the natural-language tool transcript without having to teach
  itself parsing per-tool.

Conventions for tool authors:
- Register a single template per tool with :func:`register_template`. Use
  the exact ``@mcp.tool``-registered name as the key.
- Templates should mention every placeholder exactly once (so the regex
  remains unambiguous).
- Lists / dicts in the result are flattened to comma-separated strings when
  rendered (see :func:`_stringify`). If a tool wants a different shape,
  pre-flatten the value before passing it to :func:`format_output`.
- The shared error template renders any ``error_envelope`` result, so tools
  do not need to register a second template for the failure path.
"""
from __future__ import annotations

import re
from typing import Any

# Public registry. Reward / verifier code imports this and walks it.
TOOL_TEMPLATES: dict[str, str] = {}

# Common envelope for error_envelope() returns; reused across every tool.
ERROR_TEMPLATE = "[error] tool={tool} type={error_type} message={error}"


def register_template(tool_name: str, template: str) -> str:
    """Register ``template`` for ``tool_name`` and return it unchanged.

    Returning the template lets callers use the registration as an inline
    assignment::

        _TPL = register_template("rdkit_canonicalize_smiles",
                                 "Canonical SMILES: {canonical}")
    """
    if tool_name in TOOL_TEMPLATES and TOOL_TEMPLATES[tool_name] != template:
        raise ValueError(
            f"Conflicting template for {tool_name!r}: "
            f"{TOOL_TEMPLATES[tool_name]!r} vs {template!r}"
        )
    TOOL_TEMPLATES[tool_name] = template
    return template


class _SafeDict(dict):
    """Dict that leaves unknown ``{key}`` slots literal instead of raising."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - trivial
        return "{" + key + "}"


def _stringify(value: Any) -> Any:
    """Render nested containers as flat strings for template substitution.

    Scalars (str/int/float/bool/None) pass through so format specs like
    ``{score:.3f}`` keep working. Lists become ``", "``-joined strings; dicts
    become ``"k=v, k=v"`` strings. Anything else falls back to ``repr``.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return ", ".join(
            str(_stringify(v)) if not isinstance(v, (str, int, float, bool)) else str(v)
            for v in value
        )
    if isinstance(value, tuple):
        return ", ".join(str(_stringify(v)) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}={_stringify(v)}" for k, v in value.items())
    return repr(value)


def format_output(
    result: Any,
    template: str | None,
    return_text: bool = False,
    *,
    tool_name: str | None = None,
    text_payload: dict | None = None,
) -> Any:
    """Return either the raw ``result`` (JSON mode) or a text-rendered string.

    Args:
        result: The dict-shaped result the tool would normally return.
        template: Text template registered for this tool. If ``None`` and
            ``return_text=True``, the dict is rendered as ``key=value, ...``.
        return_text: If ``False`` (default) the original ``result`` is
            returned untouched.
        tool_name: Optional name surfaced in the error template; defaults to
            ``"<unknown>"`` when omitted.
        text_payload: Optional dict used *only* for template substitution
            instead of ``result``. Use this when the natural JSON shape of
            ``result`` does not match the placeholders in ``template`` (e.g.
            a flat ``{"MW": 46.0, ...}`` dict that should render as
            ``"Descriptors: {descriptors}"``). When ``return_text=False``
            this argument is ignored.

    Errors (dicts carrying both ``error`` and ``error_type``) are rendered
    via :data:`ERROR_TEMPLATE` regardless of the tool-specific template.
    """
    if not return_text:
        return result
    if not isinstance(result, dict):
        return str(result)
    if "error" in result and "error_type" in result:
        flat = {k: _stringify(v) for k, v in result.items()}
        flat.setdefault("tool", tool_name or "<unknown>")
        return ERROR_TEMPLATE.format_map(_SafeDict(flat))
    payload = text_payload if text_payload is not None else result
    if template is None:
        return ", ".join(f"{k}={_stringify(v)}" for k, v in payload.items())
    flat = {k: _stringify(v) for k, v in payload.items()}
    return template.format_map(_SafeDict(flat))


# ---------------------------------------------------------------------------
# Reverse: template -> regex, for verifier / reward parsing.
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?::[^{}]*)?\}")


def template_placeholders(template: str) -> list[str]:
    """Return the ordered list of placeholder names in ``template``.

    Duplicate placeholders are preserved in order of appearance — useful if
    the verifier wants to detect templates that mention the same key twice.
    """
    return _PLACEHOLDER_RE.findall(template)


def template_to_regex(template: str, *, capture: str = r".*?") -> re.Pattern[str]:
    """Compile ``template`` into a regex with one named group per placeholder.

    The literal parts of the template are :func:`re.escape`-d; each ``{name}``
    (or ``{name:spec}``) becomes ``(?P<name>...)`` using ``capture`` as the
    inner pattern. The compiled regex uses :data:`re.DOTALL` so multi-line
    list renderings still match. A duplicated placeholder name becomes a
    backreference ``(?P=name)`` instead of a second group.

    The result is anchored with ``^`` / ``$`` so callers can use
    :meth:`re.Pattern.fullmatch` to parse a complete rendered tool output
    cleanly, while :meth:`~re.Pattern.search` still works on transcripts that
    embed the output among other text.
    """
    pieces: list[str] = ["^"]
    last = 0
    seen: set[str] = set()
    for m in _PLACEHOLDER_RE.finditer(template):
        pieces.append(re.escape(template[last:m.start()]))
        name = m.group(1)
        if name in seen:
            pieces.append(f"(?P={name})")
        else:
            pieces.append(f"(?P<{name}>{capture})")
            seen.add(name)
        last = m.end()
    pieces.append(re.escape(template[last:]))
    pieces.append("$")
    return re.compile("".join(pieces), re.DOTALL)


def parse_text_output(text: str, template: str) -> dict[str, str] | None:
    """Best-effort: parse a rendered text output back into a ``{name: value}`` dict.

    Returns ``None`` if the text does not match the template's shape. The
    values are returned as strings — callers should cast (``float``, ``int``,
    ``json.loads`` for list-shaped fields) as needed.
    """
    pattern = template_to_regex(template)
    m = pattern.search(text)
    if m is None:
        return None
    return m.groupdict()


__all__ = [
    "TOOL_TEMPLATES",
    "ERROR_TEMPLATE",
    "register_template",
    "format_output",
    "template_placeholders",
    "template_to_regex",
    "parse_text_output",
]
