"""Word-problem helper tools: a few happy and sad paths per tool."""

import pytest

from app.math_tools import applied as t


def call(tool, **kwargs):
    return tool.invoke(kwargs)


# ---------------------------------------------------------------- happy paths


def test_percent_all_three_modes():
    assert call(t.percent, part=18, whole=72)["percent"] == 25
    assert call(t.percent, percent=15, whole=80)["part"] == 12
    assert call(t.percent, part=30, percent=40)["whole"] == 75


def test_percent_change():
    up = call(t.percent_change, old=80, new=100)
    assert up["exact"] == "25" and up["direction"] == "increase"
    down = call(t.percent_change, old=50, new=40)
    assert down["exact"] == "-20" and down["direction"] == "decrease"


def test_ratio_proportion():
    assert call(t.ratio_proportion, a=3, b=4, c=9)["exact"] == "12"
    assert call(t.ratio_proportion, a=2, b=5, c=3)["exact"] == "15/2"


@pytest.mark.parametrize(
    "value, source, target, expected",
    [
        (5, "km", "m", 5000),
        (12, "in", "ft", 1),
        (2, "lb", "kg", 0.90718474),
        (90, "min", "h", 1.5),
        (1, "gal", "L", 3.785411784),
        (1, "m^2", "cm^2", 10000),
        (60, "mph", "km/h", 96.56064),
        (10, "m/s", "km/h", 36),
        (100, "C", "F", 212),
        (32, "F", "C", 0),
        (0, "C", "K", 273.15),
    ],
)
def test_unit_convert(value, source, target, expected):
    out = call(t.unit_convert, value=value, from_unit=source, to_unit=target)
    assert "error" not in out, out
    assert out["decimal"] == pytest.approx(expected)


def test_interest_monthly_and_continuous():
    out = call(t.simple_compound_interest, principal=1000, rate=0.05, time=3, n=12)
    assert out["compound_amount"] == 1161.47
    assert out["simple_interest"] == 150
    continuous = call(t.simple_compound_interest, principal=1000, rate=0.05, time=3, n="continuous")
    assert continuous["compound_amount"] == 1161.83
    percent_rate = call(t.simple_compound_interest, principal=1000, rate=5, time=3, n=1)
    assert percent_rate["compound_amount"] == 1157.63


# ---------------------------------------------------------------- sad paths


@pytest.mark.parametrize(
    "tool, kwargs, code",
    [
        (t.percent, {"part": 5}, "invalid_input"),
        (t.percent, {"part": 5, "whole": 0}, "math_error"),
        (t.percent_change, {"old": 0, "new": 5}, "math_error"),
        (t.ratio_proportion, {"a": 0, "b": 4, "c": 9}, "math_error"),
        (t.unit_convert, {"value": 5, "from_unit": "kg", "to_unit": "m"}, "invalid_input"),
        (t.unit_convert, {"value": 5, "from_unit": "C", "to_unit": "kg"}, "invalid_input"),
        (t.unit_convert, {"value": 5, "from_unit": "parsec", "to_unit": "m"}, "invalid_input"),
        (t.simple_compound_interest, {"principal": 1000, "rate": 0.05, "time": 3, "n": "weekly"}, "invalid_input"),
        (t.simple_compound_interest, {"principal": -5, "rate": 0.05, "time": 3}, "invalid_input"),
    ],
)
def test_sad_paths_return_error_envelope(tool, kwargs, code):
    out = call(tool, **kwargs)
    assert out.get("error") == code, out
