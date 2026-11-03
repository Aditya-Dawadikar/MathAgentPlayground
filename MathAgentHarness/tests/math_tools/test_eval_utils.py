"""The eval comparator itself needs to be trustworthy: happy and sad paths."""

import pytest

from tests.math_eval_utils import answer_assignments, answers_match, equivalent, tool_call_matches


@pytest.mark.parametrize(
    "actual, expected",
    [
        ("2*x + 3 = 11", "2x + 3 = 11"),
        ("11 = 2x + 3", "2x + 3 = 11"),
        ("4x + 6 = 22", "2x + 3 = 11"),
        ("f(x) = 3x^2 - 2", "3x^2 - 2"),
        (4.0, 4),
        ("1/2", 0.5),
        ([1, 1], [1, 1]),
        ({"x": 4, "variable": "x"}, {"x": 4}),
        ("Circle", "circle"),
        ("-x < 3", "x > -3"),
    ],
)
def test_equivalent_happy(actual, expected):
    assert equivalent(actual, expected)


@pytest.mark.parametrize(
    "actual, expected",
    [
        ("2x + 3 = 12", "2x + 3 = 11"),
        ("x > 3", "x < 3"),
        ("x^2 - 49", "x^2 + 49"),
        (5, 4),
        ([1, 2], [2, 1]),
        ({"y": 4}, {"x": 4}),
        ("square", "circle"),
        ("x = 4", "x + 4"),
    ],
)
def test_equivalent_sad(actual, expected):
    assert not equivalent(actual, expected)


def test_tool_call_matches_any_option():
    options = [{"tool": "solve_quadratic", "arguments": {"a": 1, "b": -5, "c": 6}},
               {"tool": "solve_equation", "arguments": {"equation": "x^2 - 5x + 6 = 0"}}]
    assert tool_call_matches({"name": "solve_equation", "args": {"equation": "x**2-5*x+6=0"}}, options)
    assert not tool_call_matches({"name": "factor", "args": {"expression": "x^2 - 5x + 6"}}, options)


def test_answers_match():
    assert answers_match("x = 3 or x = 2", ["2", "3"])
    assert answers_match("x = -sqrt(6)/2 - 1 or x = -1 + sqrt(6)/2", ["-1 - sqrt(6)/2", "-1 + sqrt(6)/2"])
    assert answers_match("20*pi", ["20pi"])
    assert not answers_match("x = 2", ["2", "3"])
    assert not answers_match("x = 2 or x = 4", ["2", "3"])
    assert answer_assignments("x = 6, y = 4") == {"x": "6", "y": "4"}
