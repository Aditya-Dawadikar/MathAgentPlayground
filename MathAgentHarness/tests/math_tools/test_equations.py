"""Equation solvers: a few happy and sad paths per tool."""

import pytest

from app.math_tools import equations as t


def call(tool, **kwargs):
    return tool.invoke(kwargs)


# ---------------------------------------------------------------- solve_equation


@pytest.mark.parametrize(
    "equation, solutions",
    [
        ("2x + 3 = 11", ["4"]),
        ("x^2 - 5x + 6 = 0", ["2", "3"]),
        ("3(x - 2) = 2x + 1", ["7"]),
        ("abs(x - 3) = 5", ["-2", "8"]),
        ("1/x + 1/2 = 1", ["2"]),
        ("2^x = 32", ["5"]),
    ],
)
def test_solve_equation_happy(equation, solutions):
    out = call(t.solve_equation, equation=equation)
    assert out["solution_type"] == "finite", out
    assert [str(v) for v in out["solutions"]] == solutions


def test_solve_equation_rejects_extraneous_root():
    out = call(t.solve_equation, equation="sqrt(x + 2) = x")
    assert out["solutions"] == [2]
    assert out["extraneous"] == [-1]


def test_solve_equation_rational_excluded_value():
    # x = 1 makes the denominator zero, so it is extraneous.
    out = call(t.solve_equation, equation="x/(x - 1) = 1/(x - 1)")
    assert out["solution_type"] == "none"
    assert out["extraneous"] == [1]


def test_solve_equation_identity_and_contradiction():
    assert call(t.solve_equation, equation="2(x + 1) = 2x + 2")["solution_type"] == "all_real"
    assert call(t.solve_equation, equation="x + 1 = x + 2")["solution_type"] == "none"


def test_solve_equation_no_real_solution_reports_complex():
    out = call(t.solve_equation, equation="x^2 + 1 = 0")
    assert out["solution_type"] == "none"
    assert sorted(out["complex_solutions"]) == ["-I", "I"]


# ---------------------------------------------------------------- solve_quadratic


def test_solve_quadratic_two_real_roots():
    out = call(t.solve_quadratic, a=1, b=-5, c=6)
    assert out["roots"] == [2, 3]
    assert out["solutions"] == [2, 3]  # same field name as solve_equation
    assert out["discriminant"] == 1
    assert out["nature"].startswith("two distinct real roots")


def test_solve_quadratic_non_monic_irrational():
    out = call(t.solve_quadratic, a=2, b=4, c=-1)
    assert out["roots"] == ["-sqrt(6)/2 - 1", "-1 + sqrt(6)/2"]
    assert "irrational" in out["nature"]


def test_solve_quadratic_repeated_and_complex():
    assert call(t.solve_quadratic, a=1, b=-6, c=9)["roots"] == [3]
    complex_out = call(t.solve_quadratic, a=1, b=2, c=5)
    assert complex_out["roots"] == ["-1 - 2*I", "-1 + 2*I"]
    assert complex_out["real_roots"] == []


# ---------------------------------------------------------------- solve_system


def test_solve_system_2x2_and_3x3():
    out = call(t.solve_system, equations=["x + y = 10", "x - y = 2"])
    assert out["solutions"] == [{"x": 6, "y": 4}]
    out = call(t.solve_system, equations=["x + y + z = 6", "2x - y + z = 3", "x + 2y - z = 2"])
    assert out["solutions"] == [{"x": 1, "y": 2, "z": 3}]


def test_solve_system_inconsistent_and_dependent():
    assert call(t.solve_system, equations=["x + y = 2", "x + y = 5"])["solution_type"] == "none"
    dependent = call(t.solve_system, equations=["x + y = 2", "2x + 2y = 4"])
    assert dependent["solution_type"] == "infinite"


def test_solve_system_nonlinear():
    out = call(t.solve_system, equations=["y = x^2", "y = x + 2"])
    assert sorted((s["x"], s["y"]) for s in out["solutions"]) == [(-1, 1), (2, 4)]


# ---------------------------------------------------------------- solve_inequality


@pytest.mark.parametrize(
    "inequality, interval",
    [
        ("3x - 2 <= 10", "(-inf, 4]"),
        ("-2x > 6", "(-inf, -3)"),
        ("1 < 2x + 3 <= 7", "(-1, 2]"),
        ("abs(x) < 3", "(-3, 3)"),
        ("x^2 - 4 > 0", "(-inf, -2) U (2, inf)"),
    ],
)
def test_solve_inequality_happy(inequality, interval):
    out = call(t.solve_inequality, inequality=inequality)
    assert out["interval"] == interval, out


def test_solve_inequality_flags_sign_flip():
    out = call(t.solve_inequality, inequality="-2x > 6")
    assert any("flips" in step for step in out["steps"])


def test_solve_inequality_empty_and_all():
    assert call(t.solve_inequality, inequality="x^2 < 0")["solution_type"] == "none"
    assert call(t.solve_inequality, inequality="x^2 >= 0")["solution_type"] == "all_real"


# ---------------------------------------------------------------- solve_for / verify


def test_solve_for_formula():
    assert call(t.solve_for, formula="A = 1/2*b*h", variable="h")["exact"] == "2*A/b"
    assert call(t.solve_for, formula="d = r t", variable="t")["exact"] == "d/r"


def test_verify_solution():
    assert call(t.verify_solution, equation="2x + 3 = 11", values={"x": 4})["is_solution"] is True
    assert call(t.verify_solution, equation="sqrt(x + 2) = x", values={"x": -1})["is_solution"] is False
    assert call(t.verify_solution, equation="x + y = 10", values={"x": 6, "y": 4})["is_solution"] is True
    undefined = call(t.verify_solution, equation="1/x = 2", values={"x": 0})
    assert undefined["is_solution"] is False and undefined["reason"] == "undefined at these values"


# ---------------------------------------------------------------- sad paths


@pytest.mark.parametrize(
    "tool, kwargs, code",
    [
        (t.solve_equation, {"equation": "2x + = 11"}, "parse_error"),
        (t.solve_equation, {"equation": "x + y = 3"}, "invalid_input"),
        (t.solve_equation, {"equation": "x + 1 = 3", "variable": "z"}, "invalid_input"),
        (t.solve_quadratic, {"a": 0, "b": 2, "c": 1}, "invalid_input"),
        (t.solve_system, {"equations": ["x + y = 1"]}, "invalid_input"),
        (t.solve_inequality, {"inequality": "x + 1 = 3"}, "invalid_input"),
        (t.solve_for, {"formula": "A = l w", "variable": "h"}, "invalid_input"),
        (t.verify_solution, {"equation": "x + y = 3", "values": {"x": 1}}, "invalid_input"),
    ],
)
def test_sad_paths_return_error_envelope(tool, kwargs, code):
    out = call(tool, **kwargs)
    assert out.get("error") == code, out
