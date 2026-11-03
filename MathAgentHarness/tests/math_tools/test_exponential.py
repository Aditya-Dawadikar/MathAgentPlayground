"""Exponent, log and sequence tools: a few happy and sad paths per tool."""

import pytest

from app.math_tools import exponential as t


def call(tool, **kwargs):
    return tool.invoke(kwargs)


# ---------------------------------------------------------------- happy paths


def test_log():
    assert call(t.log, x=8, base=2)["exact"] == "3"
    assert call(t.log, x=1000)["exact"] == "3"
    natural = call(t.log, x=10, base="e")
    assert natural["exact"] == "log(10)"
    assert natural["decimal"] == pytest.approx(2.302585093)


@pytest.mark.parametrize(
    "equation, solutions",
    [
        ("2^x = 32", ["5"]),
        ("3^(x+1) = 20", ["-1 + log(20)/log(3)"]),
        ("4^x = 8", ["3/2"]),
        ("log2(x) + log2(x - 2) = 3", ["4"]),
        ("log10(x) = 2", ["100"]),
    ],
)
def test_solve_exponential(equation, solutions):
    out = call(t.solve_exponential, equation=equation)
    assert [str(v) for v in out["solutions"]] == solutions, out


def test_solve_exponential_rejects_log_domain_roots():
    out = call(t.solve_exponential, equation="log2(x) + log2(x - 2) = 3")
    assert [str(v) for v in out["extraneous"]] == ["-2"]


def test_arithmetic_sequence_forward_and_backward():
    out = call(t.arithmetic_sequence, a1=3, d=4, n=10)
    assert out["nth_term"] == 39 and out["sum"] == 210
    assert call(t.arithmetic_sequence, a1=3, d=4, term_value=39)["n"] == 10
    assert call(t.arithmetic_sequence, a1=3, d=4, term_value=40)["is_term"] is False


def test_geometric_sequence():
    out = call(t.geometric_sequence, a1=2, r=3, n=5)
    assert out["nth_term"] == 162 and out["sum"] == 242
    assert call(t.geometric_sequence, a1=2, r=3, term_value=162)["n"] == 5
    converging = call(t.geometric_sequence, a1=8, r="1/2", n=3)
    assert converging["infinite_sum"] == 16 and converging["converges"] is True


def test_sequence_from_terms():
    arithmetic = call(t.sequence_from_terms, terms={"1": 2, "4": 11}, kind="arithmetic")
    assert arithmetic["d"] == 3 and arithmetic["a1"] == 2
    geometric = call(t.sequence_from_terms, terms={"2": 6, "4": 24}, kind="geometric")
    assert geometric["r"] == 2 and geometric["a1"] == 3
    assert len(geometric["options"]) == 2  # r = -2 also works
    assert call(t.sequence_from_terms, terms={"1": 3, "2": 6, "3": 12})["kind"] == "geometric"
    assert call(t.sequence_from_terms, terms={"1": 1, "2": 2, "3": 4.5})["kind"] == "neither"


# ---------------------------------------------------------------- sad paths


@pytest.mark.parametrize(
    "tool, kwargs, code",
    [
        (t.log, {"x": -5, "base": 10}, "math_error"),
        (t.log, {"x": 0}, "math_error"),
        (t.log, {"x": 8, "base": 1}, "math_error"),
        (t.solve_exponential, {"equation": "2^x = "}, "parse_error"),
        (t.arithmetic_sequence, {"a1": 1, "d": 2, "n": 0}, "invalid_input"),
        (t.geometric_sequence, {"a1": 0, "r": 2, "n": 3}, "invalid_input"),
        (t.sequence_from_terms, {"terms": {"1": 5}}, "invalid_input"),
        (t.sequence_from_terms, {"terms": {"1": 1, "5": 2}}, "invalid_input"),
    ],
)
def test_sad_paths_return_error_envelope(tool, kwargs, code):
    out = call(tool, **kwargs)
    assert out.get("error") == code, out


def test_log_equation_with_no_valid_root():
    out = call(t.solve_exponential, equation="log10(x) = log10(-x)")
    assert out["solution_type"] == "none"
