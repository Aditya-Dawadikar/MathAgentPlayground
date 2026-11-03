"""Core arithmetic tools: a few happy and sad paths per tool."""

import pytest

from app.math_tools import arithmetic as t


def call(tool, **kwargs):
    return tool.invoke(kwargs)


# ---------------------------------------------------------------- happy paths


@pytest.mark.parametrize(
    "tool, kwargs, exact",
    [
        (t.add, {"a": 2, "b": 3}, "5"),
        (t.add, {"a": "1/3", "b": "1/6"}, "1/2"),
        (t.subtract, {"a": 10, "b": 4.5}, "11/2"),
        (t.multiply, {"a": 6, "b": 7}, "42"),
        (t.divide, {"a": 1, "b": 3}, "1/3"),
        (t.power, {"base": 2, "exponent": 10}, "1024"),
        (t.power, {"base": 4, "exponent": "1/2"}, "2"),
        (t.sqrt, {"x": 72}, "6*sqrt(2)"),
        (t.nth_root, {"x": 27, "n": 3}, "3"),
        (t.nth_root, {"x": -8, "n": 3}, "-2"),
        (t.modulo, {"a": 17, "b": 5}, "2"),
        (t.absolute_value, {"x": -7.5}, "15/2"),
        (t.gcd, {"a": 48, "b": 18}, "6"),
        (t.lcm, {"a": 4, "b": 6}, "12"),
        (t.fraction_simplify, {"numerator": 42, "denominator": 56}, "3/4"),
        (t.evaluate_expression, {"expression": "(18+7)*4 - 2^3"}, "92"),
        (t.evaluate_expression, {"expression": "sqrt(16) + 3^2"}, "13"),
    ],
)
def test_happy_paths(tool, kwargs, exact):
    out = call(tool, **kwargs)
    assert "error" not in out, out
    assert str(out["exact"]) == exact


def test_divide_returns_decimal_alongside_exact():
    out = call(t.divide, a=1, b=3)
    assert out["decimal"] == pytest.approx(0.3333333333)


def test_prime_factorize_structure():
    out = call(t.prime_factorize, n=360)
    assert out["result"] == "2^3 * 3^2 * 5"
    assert out["factors"] == {"2": 3, "3": 2, "5": 1}
    assert out["is_prime"] is False


def test_even_root_of_negative_is_flagged():
    out = call(t.sqrt, x=-4)
    assert out["exact"] == "2*I"
    assert out["is_real"] is False


# ---------------------------------------------------------------- sad paths


@pytest.mark.parametrize(
    "tool, kwargs, code",
    [
        (t.divide, {"a": 5, "b": 0}, "math_error"),
        (t.modulo, {"a": 5, "b": 0}, "math_error"),
        (t.power, {"base": 0, "exponent": -1}, "math_error"),
        (t.fraction_simplify, {"numerator": 1, "denominator": 0}, "math_error"),
        (t.gcd, {"a": 4.5, "b": 3}, "invalid_input"),
        (t.prime_factorize, {"n": 1}, "invalid_input"),
        (t.nth_root, {"x": 8, "n": 0}, "invalid_input"),
        (t.add, {"a": "2x", "b": 1}, "invalid_input"),
        (t.evaluate_expression, {"expression": "2x + 1"}, "invalid_input"),
        (t.evaluate_expression, {"expression": "1/(2-2)"}, "math_error"),
        (t.evaluate_expression, {"expression": "3 +* 4"}, "parse_error"),
    ],
)
def test_sad_paths_return_error_envelope(tool, kwargs, code):
    out = call(tool, **kwargs)
    assert out.get("error") == code, out
    assert out["message"]
