"""Function tools: a few happy and sad paths per tool."""

import pytest
import sympy as sp

from app.math_tools import functions as t
from app.math_tools.core import parse_expression


def call(tool, **kwargs):
    return tool.invoke(kwargs)


# ---------------------------------------------------------------- happy paths


def test_evaluate_function():
    assert call(t.evaluate_function, expression="x^2 - 3x", value=4)["exact"] == "4"
    assert call(t.evaluate_function, expression="f(x) = 2x + 1", value="1/2")["exact"] == "2"
    assert call(t.evaluate_function, expression="g(t) = t^2", value=-3)["exact"] == "9"


def test_compose_functions():
    assert call(t.compose_functions, f="2x + 1", g="x^2")["exact"] == "2*x**2 + 1"
    assert call(t.compose_functions, f="x^2", g="2x + 1")["exact"] == "4*x**2 + 4*x + 1"


@pytest.mark.parametrize("expression", ["3x - 4", "(x + 1)/2", "x^3 + 1", "2/(x - 1)"])
def test_inverse_function_composes_to_identity(expression):
    out = call(t.inverse_function, expression=expression)
    assert out["is_one_to_one"] is True
    x = sp.Symbol("x")
    f, inverse = parse_expression(expression), parse_expression(out["exact"])
    # Compare on positive x so real cube roots simplify.
    pos = sp.Symbol("x", positive=True)
    assert sp.simplify((f.subs(x, inverse) - x).subs(x, pos + 2)) == 0


def test_inverse_of_parabola_is_not_one_to_one():
    out = call(t.inverse_function, expression="x^2 - 4x")
    assert out["is_one_to_one"] is False
    assert len(out["branches"]) == 2
    assert out["suggested_domain_restriction"] == "x >= 2"


@pytest.mark.parametrize(
    "expression, domain, range_",
    [
        ("1/(x - 2)", "(-inf, 2) U (2, inf)", "(-inf, 0) U (0, inf)"),
        ("sqrt(x - 3)", "[3, inf)", "[0, inf)"),
        ("x^2 + 1", "(-inf, inf)", "[1, inf)"),
        ("log(x)", "(0, inf)", "(-inf, inf)"),
    ],
)
def test_domain_range(expression, domain, range_):
    out = call(t.domain_range, expression=expression)
    assert out["domain"] == domain
    assert out["range"] == range_


def test_find_intercepts():
    out = call(t.find_intercepts, expression="x^2 - 4")
    assert out["x_intercepts"] == [[-2, 0], [2, 0]]
    assert out["y_intercept"] == [0, -4]
    no_y = call(t.find_intercepts, expression="1/x")
    assert no_y["x_intercepts"] == [] and no_y["y_intercept"] is None


def test_vertex_of_parabola():
    out = call(t.vertex_of_parabola, a=1, b=-4, c=3)
    assert out["vertex"] == [2, -1]
    assert out["axis_of_symmetry"] == "x = 2"
    assert out["opens"] == "up" and out["extreme_type"] == "minimum"
    assert call(t.vertex_of_parabola, a=-2, b=4, c=1)["extreme_type"] == "maximum"


# ---------------------------------------------------------------- sad paths


@pytest.mark.parametrize(
    "tool, kwargs, code",
    [
        (t.evaluate_function, {"expression": "1/(x - 2)", "value": 2}, "math_error"),
        (t.evaluate_function, {"expression": "sqrt(x)", "value": -4}, "math_error"),
        (t.evaluate_function, {"expression": "x +", "value": 1}, "parse_error"),
        (t.inverse_function, {"expression": "5"}, "invalid_input"),
        (t.vertex_of_parabola, {"a": 0, "b": 1, "c": 1}, "invalid_input"),
        (t.domain_range, {"expression": "x + y"}, "invalid_input"),
    ],
)
def test_sad_paths_return_error_envelope(tool, kwargs, code):
    out = call(tool, **kwargs)
    assert out.get("error") == code, out
