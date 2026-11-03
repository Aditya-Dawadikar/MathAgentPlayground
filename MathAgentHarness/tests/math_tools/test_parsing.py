"""Foundation tests: sandboxed parsing, envelopes, and the tool guard."""

import json
import time

import pytest
import sympy as sp

from app.math_tools.core import (
    MathToolError,
    err,
    math_tool,
    ok,
    parse_equation,
    parse_expression,
    parse_relation,
    to_number,
)

x, y = sp.symbols("x y")


# ---------------------------------------------------------------- happy paths


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2x^2 + 3(x-1)", 2 * x**2 + 3 * x - 3),
        ("xy + 2", x * y + 2),
        ("2pi", 2 * sp.pi),
        ("e^x", sp.exp(x)),
        ("abs(x - 3)", sp.Abs(x - 3)),
        ("log2(8)", sp.Integer(3)),
        ("x**2", x**2),
    ],
)
def test_parse_expression_supports_school_notation(text, expected):
    assert sp.simplify(parse_expression(text) - expected) == 0


def test_numbers_are_exact():
    assert to_number(0.05) == sp.Rational(1, 20)
    assert to_number("1/3") == sp.Rational(1, 3)
    assert to_number(4.0) == 4 and isinstance(to_number(4.0), sp.Integer)


def test_parse_relation_handles_equations_and_chained_inequalities():
    assert parse_equation("2x + 3 = 11") == sp.Eq(2 * x + 3, 11, evaluate=False)
    assert parse_equation("x^2 - 4") == sp.Eq(x**2 - 4, 0, evaluate=False)
    chained = parse_relation("1 < 2x + 3 <= 7")
    assert isinstance(chained, sp.And) and len(chained.args) == 2


def test_ok_envelope_is_json_serialisable():
    envelope = ok(sp.sqrt(8), steps=["simplified"], roots=[sp.Integer(2), sp.sqrt(3)])
    assert envelope["exact"] == "2*sqrt(2)"
    assert envelope["decimal"] == pytest.approx(2.8284271247)
    assert envelope["roots"] == [2, "sqrt(3)"]
    json.dumps(envelope)


# ---------------------------------------------------------------- sad paths


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os')",  # dunder / quotes
        "x.__class__",  # attribute access
        "open[0]",  # brackets
        "2x +",  # incomplete
        "",  # empty
    ],
)
def test_parse_expression_rejects_unsafe_or_malformed_input(text):
    with pytest.raises(MathToolError) as info:
        parse_expression(text)
    assert info.value.code in {"parse_error", "invalid_input"}


def test_python_builtins_are_unreachable():
    # 'eval' is not a known function, so it is just the product of symbols e*v*a*l.
    assert parse_expression("eval(1)").free_symbols


def test_expression_rejects_equations_and_numbers_reject_symbols():
    with pytest.raises(MathToolError):
        parse_expression("x = 2")
    with pytest.raises(MathToolError):
        to_number("2x")
    with pytest.raises(MathToolError):
        to_number(True)


def test_math_tool_converts_exceptions_to_error_envelope():
    @math_tool
    def explode(a: int) -> dict:
        """Always fails."""
        raise MathToolError("math_error", "boom")

    assert explode.invoke({"a": 1}) == err("math_error", "boom")


def test_math_tool_times_out():
    @math_tool(timeout=0.2)
    def slow(a: int) -> dict:
        """Sleeps too long."""
        time.sleep(2)
        return ok(sp.Integer(a))

    assert slow.invoke({"a": 1})["error"] == "timeout"
