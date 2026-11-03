"""Expression manipulation tools: a few happy and sad paths per tool."""

import pytest
import sympy as sp

from app.math_tools import expressions as t
from app.math_tools.core import parse_expression


def call(tool, **kwargs):
    return tool.invoke(kwargs)


def same(a: str, b: str) -> bool:
    diff = parse_expression(a) - parse_expression(b)
    # Radical tools assume non-negative variables, so compare under that assumption.
    diff = diff.subs({s: sp.Symbol(s.name, positive=True) for s in diff.free_symbols})
    return sp.simplify(diff) == 0


# ---------------------------------------------------------------- happy paths


@pytest.mark.parametrize(
    "tool, kwargs, expected",
    [
        (t.simplify, {"expression": "3x + 2 - x + 5"}, "2*x + 7"),
        (t.expand, {"expression": "(x+2)(x-3)"}, "x**2 - x - 6"),
        (t.expand, {"expression": "(2x - 1)^2"}, "4*x**2 - 4*x + 1"),
        (t.factor, {"expression": "x^2 - 9"}, "(x - 3)*(x + 3)"),
        (t.factor, {"expression": "6x^2 + 9x"}, "3*x*(2*x + 3)"),
        (t.factor, {"expression": "x^2 + 5x + 6"}, "(x + 2)*(x + 3)"),
        (t.substitute, {"expression": "x^2 + 2y", "values": {"x": 3, "y": 1}}, "11"),
        (t.rationalize_denominator, {"expression": "1/sqrt(2)"}, "sqrt(2)/2"),
        (t.rationalize_denominator, {"expression": "1/(1 + sqrt(3))"}, "(-1 + sqrt(3))/2"),
        (t.simplify_radical, {"expression": "sqrt(72)"}, "6*sqrt(2)"),
        (t.simplify_radical, {"expression": "sqrt(50x^3)"}, "5*x*sqrt(2*x)"),
        (t.combine_rational, {"expression": "1/x + 2/(x+1)"}, "(3*x + 1)/(x*(x + 1))"),
    ],
)
def test_happy_paths(tool, kwargs, expected):
    out = call(tool, **kwargs)
    assert "error" not in out, out
    assert out["exact"] == expected
    # The tool must never change the value of the expression.
    if tool is not t.substitute:
        assert same(out["exact"], kwargs["expression"])


@pytest.mark.parametrize(
    "expression, technique",
    [
        ("x^2 - 9", "difference of squares"),
        ("6x^2 + 9x", "greatest common factor 3*x"),
        ("x^2 + 6x + 9", "perfect square trinomial"),
        ("2x^2 + 7x + 3", "trinomial factoring"),
        ("x^3 + 8", "sum of cubes"),
        ("x^3 + 2x^2 + 3x + 6", "factoring by grouping"),
    ],
)
def test_factor_reports_technique(expression, technique):
    out = call(t.factor, expression=expression)
    assert technique in out["steps"], out
    assert same(out["exact"], expression)


def test_rational_simplify_reports_restrictions():
    out = call(t.simplify, expression="(x^2 - 1)/(x - 1)")
    assert out["exact"] == "x + 1"
    assert out["restrictions"] == ["x != 1"]


def test_combine_rational_restrictions():
    out = call(t.combine_rational, expression="1/x + 2/(x+1)")
    assert out["restrictions"] == ["x != -1", "x != 0"]


def test_substitute_accepts_expressions():
    out = call(t.substitute, expression="x^2", values={"x": "y + 1"})
    assert same(out["exact"], "(y + 1)^2")


# ---------------------------------------------------------------- sad paths


def test_prime_polynomial_is_not_factorable():
    out = call(t.factor, expression="x^2 + 1")
    assert out["is_factorable"] is False


@pytest.mark.parametrize(
    "tool, kwargs, code",
    [
        (t.simplify, {"expression": "2x +"}, "parse_error"),
        (t.expand, {"expression": "x = 2"}, "invalid_input"),
        (t.factor, {"expression": ""}, "parse_error"),
        (t.substitute, {"expression": "x + 1", "values": {"z": 2}}, "invalid_input"),
        (t.substitute, {"expression": "1/x", "values": {"x": 0}}, "math_error"),
        (t.substitute, {"expression": "x + 1", "values": {}}, "invalid_input"),
    ],
)
def test_sad_paths_return_error_envelope(tool, kwargs, code):
    out = call(tool, **kwargs)
    assert out.get("error") == code, out
