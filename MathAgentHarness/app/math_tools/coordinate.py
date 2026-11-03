"""Lines and coordinate-geometry tools (issue #7). Lines use the variables x and y."""

from __future__ import annotations

import sympy as sp

from .core import MathToolError, Number, math_tool, ok, parse_equation, to_number

X, Y = sp.symbols("x y")


def _point(point: list[Number], label: str) -> tuple[sp.Expr, sp.Expr]:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise MathToolError("invalid_input", f"{label} must be a point [x, y], got {point!r}")
    return to_number(point[0], f"{label} x"), to_number(point[1], f"{label} y")


def _slope_between(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    if x1 == x2 and y1 == y2:
        raise MathToolError("invalid_input", "the two points are the same, so they do not define a line")
    if x1 == x2:
        return None
    return sp.simplify((y2 - y1) / (x2 - x1))


def _standard_form(a: sp.Expr, b: sp.Expr, c: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Scale A x + B y = C to integer coefficients with A > 0 (or B > 0 when A = 0)."""
    coefficients = [sp.nsimplify(v) for v in (a, b, c)]
    if all(v.is_Rational for v in coefficients):
        scale = sp.ilcm(*[v.q for v in coefficients])
        coefficients = [v * scale for v in coefficients]
        divisor = sp.igcd(*[int(v) for v in coefficients if v != 0]) or 1
        coefficients = [v / divisor for v in coefficients]
    a, b, c = coefficients
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return a, b, c


def _line_forms(point, slope) -> dict:
    x1, y1 = point
    if slope is None:
        return {"slope": None, "slope_intercept": None, "point_slope": None, "standard": f"x = {x1}",
                "y_intercept": None, "is_vertical": True}
    intercept = sp.simplify(y1 - slope * x1)
    a, b, c = _standard_form(slope, sp.Integer(-1), -intercept)
    return {
        "slope": slope,
        "y_intercept": intercept,
        "slope_intercept": f"y = {sp.expand(slope * X + intercept)}",
        "point_slope": f"y - {y1} = {slope}*(x - {x1})".replace("- -", "+ ").replace("y - 0 =", "y =").replace("(x - 0)", "x"),
        "standard": f"{sp.expand(a * X + b * Y)} = {c}",
        "is_vertical": False,
    }


@math_tool
def slope(x1: Number, y1: Number, x2: Number, y2: Number) -> dict:
    """Slope (rise over run) of the line through (x1, y1) and (x2, y2).

    Vertical lines have undefined slope (is_vertical true); horizontal lines have slope 0.
    Example: {"x1": 1, "y1": 2, "x2": 3, "y2": 8} -> 3.
    """
    p1, p2 = (to_number(x1, "x1"), to_number(y1, "y1")), (to_number(x2, "x2"), to_number(y2, "y2"))
    m = _slope_between(p1, p2)
    if m is None:
        return ok(None, result="undefined (vertical line)", is_vertical=True, is_horizontal=False,
                  steps=[f"x1 = x2 = {p1[0]}, so the run is 0"])
    return ok(m, steps=[f"m = ({p2[1]} - {p1[1]}) / ({p2[0]} - {p1[0]})"], is_vertical=False, is_horizontal=m == 0)


@math_tool
def line_equation(point: list[Number], slope: Number | None = None, point2: list[Number] | None = None) -> dict:
    """Equation of a line from a point and a slope, or from two points.

    Returns slope-intercept, point-slope and standard (Ax + By = C) forms; vertical lines give x = k.
    Example: {"point": [1, 2], "slope": 3} -> "y = 3*x - 1", standard "3*x - y = 1";
    {"point": [0, 1], "point2": [2, 5]} -> "y = 2*x + 1".
    """
    p1 = _point(point, "point")
    if (slope is None) == (point2 is None):
        raise MathToolError("invalid_input", "give exactly one of slope or point2")
    m = to_number(slope, "slope") if slope is not None else _slope_between(p1, _point(point2, "point2"))
    forms = _line_forms(p1, m)
    result = forms["slope_intercept"] or forms["standard"]
    return ok(None, result=result, **forms)


@math_tool
def distance(point1: list[Number], point2: list[Number]) -> dict:
    """Distance between two points, in exact radical form plus decimal.

    Example: {"point1": [0, 0], "point2": [3, 4]} -> 5; {"point1": [1, 2], "point2": [3, 6]} -> "2*sqrt(5)".
    """
    (x1, y1), (x2, y2) = _point(point1, "point1"), _point(point2, "point2")
    value = sp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return ok(value, steps=[f"d = sqrt(({x2} - {x1})^2 + ({y2} - {y1})^2)"])


@math_tool
def midpoint(point1: list[Number], point2: list[Number]) -> dict:
    """Midpoint of the segment between two points.

    Example: {"point1": [2, 4], "point2": [6, 8]} -> [4, 6].
    """
    (x1, y1), (x2, y2) = _point(point1, "point1"), _point(point2, "point2")
    mid = [(x1 + x2) / 2, (y1 + y2) / 2]
    envelope = ok(None, result=f"({mid[0]}, {mid[1]})", midpoint=mid)
    envelope["exact"] = envelope["midpoint"]
    return envelope


def _line_coefficients(text: str, label: str):
    if "=" not in str(text):
        text = f"y = {text}"
    eq = parse_equation(text, label)
    expr = sp.expand(eq.lhs - eq.rhs)
    if expr.free_symbols - {X, Y}:
        raise MathToolError("invalid_input", f"{label} must only use the variables x and y")
    try:
        poly = sp.Poly(expr, X, Y)
    except sp.PolynomialError as exc:
        raise MathToolError("invalid_input", f"{label} is not a line: {text!r}") from exc
    if poly.total_degree() != 1:
        raise MathToolError("invalid_input", f"{label} is not a line: {text!r}")
    return expr.coeff(X), expr.coeff(Y), expr.subs({X: 0, Y: 0})


@math_tool
def parallel_or_perpendicular(line1: str, line2: str) -> dict:
    """Classify two lines as parallel, perpendicular, the same line, or just intersecting.

    Lines can be in any form ("y = 2x + 1", "2x - y = 3", "x = 4").
    Example: {"line1": "y = 2x + 1", "line2": "y = -1/2x + 3"} -> "perpendicular".
    """
    a1, b1, c1 = _line_coefficients(line1, "line1")
    a2, b2, c2 = _line_coefficients(line2, "line2")
    slopes = [None if b == 0 else sp.simplify(-a / b) for a, b in ((a1, b1), (a2, b2))]
    intersection = None
    if sp.simplify(a1 * b2 - a2 * b1) == 0:
        same = sp.simplify(a1 * c2 - a2 * c1) == 0 and sp.simplify(b1 * c2 - b2 * c1) == 0
        relationship = "same line" if same else "parallel"
    else:
        relationship = "perpendicular" if sp.simplify(a1 * a2 + b1 * b2) == 0 else "intersecting"
        solution = sp.solve([a1 * X + b1 * Y + c1, a2 * X + b2 * Y + c2], [X, Y], dict=True)[0]
        intersection = [solution[X], solution[Y]]
    return ok(None, result=relationship, relationship=relationship, slopes=slopes, intersection=intersection)


TOOLS = [slope, line_equation, distance, midpoint, parallel_or_perpendicular]
