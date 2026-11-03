"""Coordinate-geometry tools: a few happy and sad paths per tool."""

import pytest

from app.math_tools import coordinate as t


def call(tool, **kwargs):
    return tool.invoke(kwargs)


# ---------------------------------------------------------------- happy paths


def test_slope_regular_horizontal_vertical():
    assert call(t.slope, x1=1, y1=2, x2=3, y2=8)["exact"] == "3"
    assert call(t.slope, x1=0, y1=5, x2=4, y2=5)["is_horizontal"] is True
    vertical = call(t.slope, x1=2, y1=1, x2=2, y2=9)
    assert vertical["is_vertical"] is True and vertical["exact"] is None


def test_line_equation_point_slope():
    out = call(t.line_equation, point=[1, 2], slope=3)
    assert out["slope_intercept"] == "y = 3*x - 1"
    assert out["standard"] == "3*x - y = 1"
    assert out["point_slope"] == "y - 2 = 3*(x - 1)"


def test_line_equation_two_points_and_fractional_slope():
    assert call(t.line_equation, point=[0, 1], point2=[2, 5])["slope_intercept"] == "y = 2*x + 1"
    out = call(t.line_equation, point=[0, 2], slope="-2/3")
    assert out["standard"] == "2*x + 3*y = 6"


def test_line_equation_vertical():
    out = call(t.line_equation, point=[4, 1], point2=[4, 7])
    assert out["is_vertical"] is True
    assert out["result"] == "x = 4"


def test_distance_and_midpoint():
    assert call(t.distance, point1=[0, 0], point2=[3, 4])["exact"] == "5"
    radical = call(t.distance, point1=[1, 2], point2=[3, 6])
    assert radical["exact"] == "2*sqrt(5)"
    assert radical["decimal"] == pytest.approx(4.472135955)
    assert call(t.midpoint, point1=[2, 4], point2=[6, 8])["midpoint"] == [4, 6]
    assert call(t.midpoint, point1=[1, 2], point2=[4, 7])["midpoint"] == ["5/2", "9/2"]


@pytest.mark.parametrize(
    "line1, line2, relationship",
    [
        ("y = 2x + 1", "y = -1/2x + 3", "perpendicular"),
        ("y = 3x + 1", "6x - 2y = 10", "parallel"),
        ("2x + y = 4", "y = -2x + 4", "same line"),
        ("x = 4", "y = 1", "perpendicular"),
        ("y = x", "y = 2x - 3", "intersecting"),
    ],
)
def test_parallel_or_perpendicular(line1, line2, relationship):
    assert call(t.parallel_or_perpendicular, line1=line1, line2=line2)["relationship"] == relationship


def test_intersection_point_reported():
    out = call(t.parallel_or_perpendicular, line1="y = x", line2="y = 2x - 3")
    assert out["intersection"] == [3, 3]


# ---------------------------------------------------------------- sad paths


@pytest.mark.parametrize(
    "tool, kwargs, code",
    [
        (t.slope, {"x1": 1, "y1": 1, "x2": 1, "y2": 1}, "invalid_input"),
        (t.line_equation, {"point": [1, 2]}, "invalid_input"),
        (t.line_equation, {"point": [1, 2], "slope": 3, "point2": [3, 4]}, "invalid_input"),
        (t.distance, {"point1": [1, 2, 3], "point2": [0, 0]}, "invalid_input"),
        (t.parallel_or_perpendicular, {"line1": "y = x^2", "line2": "y = x"}, "invalid_input"),
        (t.parallel_or_perpendicular, {"line1": "y = z + 1", "line2": "y = x"}, "invalid_input"),
    ],
)
def test_sad_paths_return_error_envelope(tool, kwargs, code):
    out = call(tool, **kwargs)
    assert out.get("error") == code, out
