"""Geometry and trig tools: a few happy and sad paths per tool."""

import pytest

from app.math_tools import geometry as t


def call(tool, **kwargs):
    return tool.invoke(kwargs)


# ---------------------------------------------------------------- happy paths


def test_pythagorean():
    assert call(t.pythagorean, a=3, b=4)["exact"] == "5"
    assert call(t.pythagorean, a=5, c=13)["exact"] == "12"
    assert call(t.pythagorean, a=1, b=1)["exact"] == "sqrt(2)"


@pytest.mark.parametrize(
    "tool, shape, dims, exact",
    [
        (t.area, "circle", {"r": 3}, "9*pi"),
        (t.area, "circle", {"diameter": 10}, "25*pi"),
        (t.area, "triangle", {"b": 6, "h": 4}, "12"),
        (t.area, "triangle", {"a": 3, "b": 4, "c": 5}, "6"),
        (t.area, "trapezoid", {"b1": 3, "b2": 5, "h": 4}, "16"),
        (t.perimeter, "rectangle", {"l": 5, "w": 3}, "16"),
        (t.perimeter, "circle", {"r": 3}, "6*pi"),
        (t.volume, "cylinder", {"r": 2, "h": 5}, "20*pi"),
        (t.volume, "sphere", {"r": 3}, "36*pi"),
        (t.volume, "cone", {"r": 3, "h": 4}, "12*pi"),
        (t.volume, "rectangular prism", {"length": 2, "width": 3, "height": 4}, "24"),
        (t.surface_area, "cube", {"s": 3}, "54"),
        (t.surface_area, "cone", {"r": 3, "h": 4}, "24*pi"),
    ],
)
def test_measurements(tool, shape, dims, exact):
    out = call(tool, shape=shape, dimensions=dims)
    assert "error" not in out, out
    assert out["exact"] == exact


@pytest.mark.parametrize(
    "function, angle, exact",
    [("sin", 30, "1/2"), ("cos", 45, "sqrt(2)/2"), ("tan", 60, "sqrt(3)"), ("cos", 90, "0")],
)
def test_trig_special_angles(function, angle, exact):
    assert call(t.trig, function=function, angle=angle)["exact"] == exact


def test_trig_radians_and_inverse():
    assert call(t.trig, function="sin", angle="pi/2", unit="radians")["exact"] == "1"
    assert call(t.inverse_trig, function="tan", value=1)["exact"] == "45"
    assert call(t.inverse_trig, function="arcsin", value="1/2")["exact"] == "30"
    assert call(t.inverse_trig, function="acos", value=0.2)["decimal"] == pytest.approx(78.463, abs=1e-3)


def test_solve_right_triangle_from_sides_and_angle():
    out = call(t.solve_right_triangle, a=3, b=4)
    assert out["sides"]["c"] == 5
    assert out["decimals"]["A"] == pytest.approx(36.8699)
    special = call(t.solve_right_triangle, c=10, A=30)
    assert special["sides"] == {"a": 5, "b": "5*sqrt(3)", "c": 10}
    assert special["angles"]["B"] == 60


# ---------------------------------------------------------------- sad paths


@pytest.mark.parametrize(
    "tool, kwargs, code",
    [
        (t.pythagorean, {"a": 3}, "invalid_input"),
        (t.pythagorean, {"a": 5, "c": 4}, "invalid_input"),
        (t.area, {"shape": "hexagon", "dimensions": {"s": 2}}, "invalid_input"),
        (t.area, {"shape": "rectangle", "dimensions": {"l": 5}}, "invalid_input"),
        (t.area, {"shape": "triangle", "dimensions": {"a": 1, "b": 2, "c": 5}}, "invalid_input"),
        (t.volume, {"shape": "sphere", "dimensions": {"r": -2}}, "invalid_input"),
        (t.trig, {"function": "tan", "angle": 90}, "math_error"),
        (t.inverse_trig, {"function": "sin", "value": 2}, "math_error"),
        (t.solve_right_triangle, {"A": 30, "B": 60}, "invalid_input"),
        (t.solve_right_triangle, {"a": 3, "A": 95}, "invalid_input"),
    ],
)
def test_sad_paths_return_error_envelope(tool, kwargs, code):
    out = call(tool, **kwargs)
    assert out.get("error") == code, out


def test_missing_dimension_error_lists_requirements():
    out = call(t.area, shape="rectangle", dimensions={"l": 5})
    assert "{l, w}" in out["message"]
