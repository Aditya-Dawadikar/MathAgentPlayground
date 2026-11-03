"""Semantic comparison helpers for the math agent evals.

LLMs write the same math many ways ("2x+3=11", "2*x + 3 = 11", "f(x) = ..."),
so tool arguments and final answers are compared by meaning, not by string.
"""

from __future__ import annotations

import re
from typing import Any

import sympy as sp

from app.math_tools.core import MathToolError, parse_expression, parse_relation

_DEFINITION = re.compile(r"^\s*(?:[A-Za-z]\s*\(\s*[A-Za-z]\s*\)|y)\s*=\s*")


def _parse(value: Any):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return parse_expression(value)
    if not isinstance(value, str):
        return None
    text = _DEFINITION.sub("", value.replace("$", "").strip())
    try:
        if re.search(r"[=<>]", text):
            return parse_relation(text)
        return parse_expression(text)
    except MathToolError:
        return None


def _relations_equivalent(a, b) -> bool:
    if isinstance(a, sp.And) != isinstance(b, sp.And) or isinstance(a, sp.Equality) != isinstance(b, sp.Equality):
        return False
    if isinstance(a, sp.And):
        return len(a.args) == len(b.args) and all(any(_relations_equivalent(x, y) for y in b.args) for x in a.args)
    left, right = a.lhs - a.rhs, b.lhs - b.rhs
    if isinstance(a, sp.Equality):
        if sp.simplify(left - right) == 0 or sp.simplify(left + right) == 0:
            return True
        ratio = sp.simplify(left / right) if right != 0 else None
        return bool(ratio is not None and ratio.is_number and ratio != 0)
    (op_a, left), (op_b, right) = _canonical(a), _canonical(b)
    if op_a != op_b:
        return False
    ratio = sp.simplify(left / right) if right != 0 else None
    return bool(ratio is not None and ratio.is_number and ratio.is_positive)


def _canonical(relation):
    """Rewrite as (op, expr) meaning ``expr op 0`` with op in {"<", "<="}."""
    expr = relation.lhs - relation.rhs
    if relation.rel_op in (">", ">="):
        return relation.rel_op.replace(">", "<"), -expr
    return relation.rel_op, expr


def equivalent(actual: Any, expected: Any) -> bool:
    """True if two tool arguments / answers mean the same thing."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and equivalent(actual[k], v) for k, v in expected.items())
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(
            equivalent(a, e) for a, e in zip(actual, expected)
        )
    if isinstance(expected, str) and not re.search(r"\d|[=<>^*/+\-()]", expected):
        # Plain words such as a shape or unit name.
        return isinstance(actual, str) and actual.strip().lower().replace(" ", "_") == expected.strip().lower().replace(" ", "_")
    a, e = _parse(actual), _parse(expected)
    if a is None or e is None:
        return str(actual).strip() == str(expected).strip()
    if isinstance(e, sp.Basic) and isinstance(e, (sp.Equality, sp.core.relational.Relational, sp.And)):
        return isinstance(a, (sp.Equality, sp.core.relational.Relational, sp.And)) and _relations_equivalent(a, e)
    if isinstance(a, (sp.Equality, sp.core.relational.Relational, sp.And)):
        return False
    try:
        return sp.simplify(a - e) == 0
    except TypeError:
        return False


def tool_call_matches(call: dict, expected_options: list[dict]) -> bool:
    """A call matches when its name and the expected (partial) arguments match any option."""
    return any(
        call["name"] == option["tool"] and equivalent(call.get("args", {}), option.get("arguments", {}))
        for option in expected_options
    )


def answer_values(text: Any) -> list[str]:
    """Extract answer values from "x = 2 or x = 3", "x = 1, x = 2", "40", "20*pi"."""
    if not isinstance(text, str):
        return [str(text)]
    parts = re.split(r"\s+or\s+|,\s*", text.strip())
    values = []
    for part in parts:
        part = part.strip().strip("()")
        if not part:
            continue
        values.append(part.split("=", 1)[1].strip() if "=" in part else part)
    return values


def answer_assignments(text: str) -> dict[str, str]:
    """Extract {"x": "6", "y": "4"} from "x = 6, y = 4"."""
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"([A-Za-z]\w*)\s*=\s*([^,]+)", text or "")}


def answers_match(final_output: Any, expected: list[Any]) -> bool:
    """Order-insensitive semantic comparison of answer values."""
    remaining = answer_values(final_output)
    if len(remaining) != len(expected):
        return False
    for value in expected:
        match = next((r for r in remaining if equivalent(r, value)), None)
        if match is None:
            return False
        remaining.remove(match)
    return True


def contains_any(text: Any, fragments: list[str]) -> bool:
    lowered = str(text).lower()
    return any(fragment.lower() in lowered for fragment in fragments)
