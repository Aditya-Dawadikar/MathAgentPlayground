"""SymPy-backed math tools for the agents.

See :mod:`app.math_tools.core` for the shared parsing rules and the
result/error envelope every tool returns.
"""

from . import arithmetic, coordinate, equations, exponential, expressions, functions, polynomials
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
}

ALL_TOOLS = [tool for group in TOOL_GROUPS.values() for tool in group]
TOOLS_BY_NAME = {tool.name: tool for tool in ALL_TOOLS}

__all__ = [
    "ALL_TOOLS",
    "TOOLS_BY_NAME",
    "TOOL_GROUPS",
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
