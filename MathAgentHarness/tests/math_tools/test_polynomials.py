"""Polynomial tools: a few happy and sad paths per tool."""

import pytest
import sympy as sp

from app.math_tools import polynomials as t
from app.math_tools.core import parse_expression


def call(tool, **kwargs):
    return tool.invoke(kwargs)


# ---------------------------------------------------------------- happy paths


def test_polynomial_degree():
    out = call(t.polynomial_degree, expression="4x^3 - x + 7")
    assert out["degree"] == 3
    assert out["leading_coefficient"] == 4
    assert out["name"] == "cubic trinomial"


@pytest.mark.parametrize(
    "dividend, divisor, quotient, remainder, method",
    [
        ("x^3 - 1", "x - 1", "x**2 + x + 1", 0, "synthetic division"),
        ("2x^3 + 3x^2 - x + 5", "x + 2", "2*x**2 - x + 1", 3, "synthetic division"),
        ("x^4 + 1", "x^2 + 1", "x**2 - 1", 2, "long division"),
    ],
)
def test_polynomial_divide(dividend, divisor, quotient, remainder, method):
    out = call(t.polynomial_divide, dividend=dividend, divisor=divisor)
    assert out["quotient"] == quotient
    assert out["remainder"] == remainder
    assert out["method"] == method
    # quotient * divisor + remainder == dividend
    q, d, r, p = (parse_expression(str(v)) for v in (quotient, divisor, remainder, dividend))
    assert sp.expand(q * d + r - p) == 0


def test_find_roots_with_multiplicity_and_complex():
    out = call(t.find_roots, polynomial="x^3 - 6x^2 + 11x - 6")
    assert out["real_roots"] == [1, 2, 3]
    repeated = call(t.find_roots, polynomial="(x - 2)^2 (x + 1)")
    assert {r["value"]: r["multiplicity"] for r in repeated["roots"]} == {-1: 1, 2: 2}
    mixed = call(t.find_roots, polynomial="x^3 - x^2 + x - 1")
    assert mixed["real_roots"] == [1]
    assert sorted(mixed["complex_roots"]) == ["-I", "I"]


@pytest.mark.parametrize(
    "expression, vertex_form, vertex",
    [
        ("x^2 + 6x + 5", "(x + 3)**2 - 4", [-3, -4]),
        ("2x^2 + 8x + 3", "2*(x + 2)**2 - 5", [-2, -5]),
        ("x^2 - 3x", "(x - 3/2)**2 - 9/4", ["3/2", "-9/4"]),
    ],
)
def test_complete_the_square(expression, vertex_form, vertex):
    out = call(t.complete_the_square, expression=expression)
    assert out["exact"] == vertex_form
    assert out["vertex"] == vertex
    assert sp.expand(parse_expression(out["exact"]) - parse_expression(expression)) == 0


def test_discriminant():
    assert call(t.discriminant, a=1, b=2, c=5)["real_root_count"] == 0
    assert call(t.discriminant, a=1, b=-4, c=4)["nature"] == "one repeated real root"
    out = call(t.discriminant, a=1, b=-5, c=6)
    assert out["exact"] == "1" and out["rational_roots"] is True


# ---------------------------------------------------------------- sad paths


@pytest.mark.parametrize(
    "tool, kwargs, code",
    [
        (t.polynomial_degree, {"expression": "1/x + 2"}, "invalid_input"),
        (t.polynomial_degree, {"expression": "sqrt(x) + 1"}, "invalid_input"),
        (t.polynomial_divide, {"dividend": "x^2 + 1", "divisor": "0"}, "math_error"),
        (t.find_roots, {"polynomial": "7"}, "invalid_input"),
        (t.complete_the_square, {"expression": "x^3 + x"}, "invalid_input"),
        (t.discriminant, {"a": 0, "b": 1, "c": 1}, "invalid_input"),
    ],
)
def test_sad_paths_return_error_envelope(tool, kwargs, code):
    out = call(tool, **kwargs)
    assert out.get("error") == code, out
