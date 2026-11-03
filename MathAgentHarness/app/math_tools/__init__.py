"""SymPy-backed math tools for the agents.

See :mod:`app.math_tools.core` for the shared parsing rules and the
result/error envelope every tool returns.
"""

from . import (
    applied,
    arithmetic,
    coordinate,
    equations,
    exponential,
    expressions,
    functions,
    geometry,
    polynomials,
    utility,
)
from .core import (
    MathToolError,
    Number,
    err,
    math_tool,
    ok,
    parse_equation,
    parse_expression,
    parse_relation,
    parse_variable,
    to_number,
)

# Populated by each tool category module.
TOOL_GROUPS: dict[str, list] = {
    "arithmetic": arithmetic.TOOLS,
    "expressions": expressions.TOOLS,
    "equations": equations.TOOLS,
    "polynomials": polynomials.TOOLS,
    "functions": functions.TOOLS,
    "coordinate": coordinate.TOOLS,
    "exponential": exponential.TOOLS,
    "applied": applied.TOOLS,
    "geometry": geometry.TOOLS,
    "utility": utility.TOOLS,
}

ALL_TOOLS = [tool for group in TOOL_GROUPS.values() for tool in group]
TOOLS_BY_NAME = {tool.name: tool for tool in ALL_TOOLS}
TOOL_NAMES = tuple(TOOLS_BY_NAME)


def _signature(tool) -> str:
    schema = tool.args_schema.model_json_schema() if tool.args_schema else {}
    required = set(schema.get("required", []))
    params = [name if name in required else f"{name}?" for name in schema.get("properties", {})]
    return f"{tool.name}({', '.join(params)})"


def describe_tools(tools=None) -> str:
    """Compact catalog for prompts: one line per tool, grouped by category.

    Example line: ``- solve_equation(equation, variable?): Solve a single equation ...``
    """
    selected = {t.name for t in tools} if tools is not None else None
    lines = []
    for group, group_tools in TOOL_GROUPS.items():
        entries = [t for t in group_tools if selected is None or t.name in selected]
        if not entries:
            continue
        lines.append(f"[{group} tools - the heading is not a tool name]")
        for t in entries:
            summary = (t.description or "").strip().splitlines()[0]
            lines.append(f"- {_signature(t)}: {summary}")
    return "\n".join(lines)


def suggest_tools(name: str, limit: int = 5) -> list[str]:
    """Valid tool names for a wrong one: a category's tools, or close spellings."""
    import difflib

    key = (name or "").strip().lower()
    if key in TOOL_GROUPS:
        return [t.name for t in TOOL_GROUPS[key]]
    return difflib.get_close_matches(key, TOOL_NAMES, n=limit, cutoff=0.5)


__all__ = [
    "suggest_tools",
    "ALL_TOOLS",
    "TOOLS_BY_NAME",
    "TOOL_GROUPS",
    "TOOL_NAMES",
    "describe_tools",
    "MathToolError",
    "Number",
    "err",
    "math_tool",
    "ok",
    "parse_equation",
    "parse_expression",
    "parse_relation",
    "parse_variable",
    "to_number",
]
