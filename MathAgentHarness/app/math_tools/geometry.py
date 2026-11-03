"""Geometry and right-triangle trigonometry tools (issue #10). Angles default to degrees."""

from __future__ import annotations

from typing import Callable

import sympy as sp

from .core import MathToolError, Number, math_tool, ok, to_number

pi = sp.pi

# measure -> shape -> list of (required dimension names, formula)
_Formula = tuple[tuple[str, ...], Callable[..., sp.Expr]]
_FORMULAS: dict[str, dict[str, list[_Formula]]] = {
    "area": {
        "square": [(("s",), lambda s: s**2)],
        "rectangle": [(("l", "w"), lambda l, w: l * w)],
        "triangle": [(("b", "h"), lambda b, h: b * h / 2),
                     (("a", "b", "c"), lambda a, b, c: sp.sqrt(((a + b + c) / 2) * ((b + c - a) / 2) * ((a + c - b) / 2) * ((a + b - c) / 2)))],
        "circle": [(("r",), lambda r: pi * r**2)],
        "trapezoid": [(("b1", "b2", "h"), lambda b1, b2, h: (b1 + b2) * h / 2)],
        "parallelogram": [(("b", "h"), lambda b, h: b * h)],
    },
    "perimeter": {
        "square": [(("s",), lambda s: 4 * s)],
        "rectangle": [(("l", "w"), lambda l, w: 2 * (l + w))],
        "triangle": [(("a", "b", "c"), lambda a, b, c: a + b + c)],
        "circle": [(("r",), lambda r: 2 * pi * r)],
        "trapezoid": [(("b1", "b2", "l1", "l2"), lambda b1, b2, l1, l2: b1 + b2 + l1 + l2)],
        "parallelogram": [(("b", "s"), lambda b, s: 2 * (b + s))],
    },
    "volume": {
        "cube": [(("s",), lambda s: s**3)],
        "rectangular_prism": [(("l", "w", "h"), lambda l, w, h: l * w * h)],
        "cylinder": [(("r", "h"), lambda r, h: pi * r**2 * h)],
        "cone": [(("r", "h"), lambda r, h: pi * r**2 * h / 3)],
        "sphere": [(("r",), lambda r: sp.Rational(4, 3) * pi * r**3)],
        "square_pyramid": [(("s", "h"), lambda s, h: s**2 * h / 3)],
    },
    "surface_area": {
        "cube": [(("s",), lambda s: 6 * s**2)],
        "rectangular_prism": [(("l", "w", "h"), lambda l, w, h: 2 * (l * w + l * h + w * h))],
        "cylinder": [(("r", "h"), lambda r, h: 2 * pi * r**2 + 2 * pi * r * h)],
        "cone": [(("r", "h"), lambda r, h: pi * r**2 + pi * r * sp.sqrt(r**2 + h**2))],
        "sphere": [(("r",), lambda r: 4 * pi * r**2)],
        "square_pyramid": [(("s", "h"), lambda s, h: s**2 + 2 * s * sp.sqrt((s / 2) ** 2 + h**2))],
    },
}
_SHAPE_ALIASES = {"rectangular prism": "rectangular_prism", "box": "rectangular_prism", "prism": "rectangular_prism",
                  "pyramid": "square_pyramid", "square pyramid": "square_pyramid"}
_DIM_ALIASES = {"side": "s", "length": "l", "width": "w", "height": "h", "base": "b", "radius": "r",
                "diameter": "d", "base1": "b1", "base2": "b2"}


def _measure(measure: str, shape: str, dimensions: dict[str, Number]) -> dict:
    key = shape.strip().lower()
    key = _SHAPE_ALIASES.get(key, key.replace(" ", "_"))
    shapes = _FORMULAS[measure]
    if key not in shapes:
        raise MathToolError("invalid_input", f"unsupported shape {shape!r} for {measure}; supported: {', '.join(sorted(shapes))}")
    dims = {}
    for name, value in (dimensions or {}).items():
        dim = _DIM_ALIASES.get(name.strip().lower(), name.strip())
        number = to_number(value, name)
        if not number.is_positive:
            raise MathToolError("invalid_input", f"dimension {name} must be positive, got {value}")
        dims[dim] = number
    if "d" in dims and "r" not in dims:
        dims["r"] = dims.pop("d") / 2
    for required, formula in shapes[key]:
        if all(r in dims for r in required):
            args = [dims[r] for r in required]
            if key == "triangle" and len(required) == 3:
                a, b, c = args
                if not (a + b > c and a + c > b and b + c > a):
                    raise MathToolError("invalid_input", f"sides {a}, {b}, {c} violate the triangle inequality")
            value = sp.simplify(formula(*args))
            return ok(value, shape=key, formula_inputs=dict(zip(required, args)))
    options = " or ".join("{" + ", ".join(r) + "}" for r, _ in shapes[key])
    raise MathToolError("invalid_input", f"{measure} of a {key} needs dimensions {options}")


@math_tool
def area(shape: str, dimensions: dict[str, Number]) -> dict:
    """Area of a 2D shape: square {s}, rectangle {l, w}, triangle {b, h} or {a, b, c},
    circle {r} or {d}, trapezoid {b1, b2, h}, parallelogram {b, h}.

    Example: {"shape": "circle", "dimensions": {"r": 3}} -> "9*pi" (28.27...).
    """
    return _measure("area", shape, dimensions)


@math_tool
def perimeter(shape: str, dimensions: dict[str, Number]) -> dict:
    """Perimeter (circumference for circles): square {s}, rectangle {l, w}, triangle {a, b, c},
    circle {r} or {d}, trapezoid {b1, b2, l1, l2}, parallelogram {b, s}.

    Example: {"shape": "rectangle", "dimensions": {"l": 5, "w": 3}} -> 16.
    """
    return _measure("perimeter", shape, dimensions)


@math_tool
def volume(shape: str, dimensions: dict[str, Number]) -> dict:
    """Volume of a solid: cube {s}, rectangular_prism {l, w, h}, cylinder {r, h}, cone {r, h},
    sphere {r}, square_pyramid {s, h}. Radius can be given as diameter d.

    Example: {"shape": "cylinder", "dimensions": {"r": 2, "h": 5}} -> "20*pi".
    """
    return _measure("volume", shape, dimensions)


@math_tool
def surface_area(shape: str, dimensions: dict[str, Number]) -> dict:
    """Total surface area of a solid: cube {s}, rectangular_prism {l, w, h}, cylinder {r, h},
    cone {r, h}, sphere {r}, square_pyramid {s, h}.

    Example: {"shape": "cube", "dimensions": {"s": 3}} -> 54.
    """
    return _measure("surface_area", shape, dimensions)


@math_tool
def pythagorean(a: Number | None = None, b: Number | None = None, c: Number | None = None) -> dict:
    """Pythagorean theorem a^2 + b^2 = c^2 (c is the hypotenuse): give any two sides to get the third.

    Example: {"a": 3, "b": 4} -> c = 5; {"a": 5, "c": 13} -> b = 12.
    """
    given = {k: to_number(v, k) for k, v in (("a", a), ("b", b), ("c", c)) if v is not None}
    if len(given) != 2:
        raise MathToolError("invalid_input", "give exactly two of a, b (legs) and c (hypotenuse)")
    if any(not v.is_positive for v in given.values()):
        raise MathToolError("invalid_input", "side lengths must be positive")
    if "c" not in given:
        missing, value = "c", sp.sqrt(given["a"] ** 2 + given["b"] ** 2)
    else:
        leg = given.get("a", given.get("b"))
        if given["c"] <= leg:
            raise MathToolError("invalid_input", "the hypotenuse c must be longer than either leg")
        missing, value = ("b" if "a" in given else "a"), sp.sqrt(given["c"] ** 2 - leg**2)
    return ok(value, result=f"{missing} = {value}", missing_side=missing)


_TRIG = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "csc": sp.csc, "sec": sp.sec, "cot": sp.cot}
_INVERSE = {"sin": sp.asin, "cos": sp.acos, "tan": sp.atan}


def _to_radians(angle: sp.Expr, unit: str) -> sp.Expr:
    unit = unit.strip().lower()
    if unit in ("degrees", "degree", "deg"):
        return angle * pi / 180
    if unit in ("radians", "radian", "rad"):
        return angle
    raise MathToolError("invalid_input", "unit must be 'degrees' or 'radians'")


@math_tool
def trig(function: str, angle: Number, unit: str = "degrees") -> dict:
    """Evaluate sin, cos, tan, csc, sec or cot of an angle (degrees by default), exactly for special angles.

    Example: {"function": "sin", "angle": 30} -> "1/2"; {"function": "cos", "angle": 45} -> "sqrt(2)/2".
    """
    name = function.strip().lower()
    if name not in _TRIG:
        raise MathToolError("invalid_input", f"function must be one of {', '.join(_TRIG)}")
    value = sp.simplify(_TRIG[name](_to_radians(to_number(angle, "angle"), unit)))
    if value.has(sp.zoo, sp.nan):
        raise MathToolError("math_error", f"{name}({angle}) is undefined")
    return ok(value)


@math_tool
def inverse_trig(function: str, value: Number, unit: str = "degrees") -> dict:
    """Inverse trig: the angle whose sin/cos/tan is value (accepts "sin", "asin" or "arcsin").

    Returns degrees by default. Example: {"function": "tan", "value": 1} -> 45;
    {"function": "sin", "value": "1/2"} -> 30.
    """
    name = function.strip().lower().removeprefix("arc").removeprefix("a")
    if name not in _INVERSE:
        raise MathToolError("invalid_input", "function must be sin, cos or tan (or asin, acos, atan)")
    x = to_number(value, "value")
    if name in ("sin", "cos") and not (-1 <= x <= 1):
        raise MathToolError("math_error", f"arc{name} is only defined for values between -1 and 1, got {x}")
    radians = _INVERSE[name](x)
    angle = sp.simplify(radians * 180 / pi) if unit.strip().lower().startswith("deg") else _to_radians(radians, unit)
    return ok(angle, result=f"{angle}" + (" degrees" if unit.strip().lower().startswith("deg") else " radians"))


@math_tool
def solve_right_triangle(a: Number | None = None, b: Number | None = None, c: Number | None = None,
                         A: Number | None = None, B: Number | None = None) -> dict:
    """Solve a right triangle (C = 90 degrees) from two known values, at least one a side.

    a and b are legs, c is the hypotenuse, A is the angle opposite a and B opposite b (degrees).
    Example: {"a": 3, "b": 4} -> c = 5, A = 36.87, B = 53.13; {"c": 10, "A": 30} -> a = 5, b = 5*sqrt(3).
    """
    sides = {k: to_number(v, k) for k, v in (("a", a), ("b", b), ("c", c)) if v is not None}
    angles = {k: to_number(v, k) for k, v in (("A", A), ("B", B)) if v is not None}
    if not sides:
        raise MathToolError("invalid_input", "at least one side is needed (angles alone fix the shape, not the size)")
    if len(sides) + (1 if angles else 0) < 2:
        raise MathToolError("invalid_input", "give two known values, at least one of them a side")
    if any(not v.is_positive for v in sides.values()):
        raise MathToolError("invalid_input", "side lengths must be positive")
    if any(not (0 < v < 90) for v in angles.values()):
        raise MathToolError("invalid_input", "acute angles must be strictly between 0 and 90 degrees")
    if "c" in sides and any(sides["c"] <= sides[k] for k in ("a", "b") if k in sides):
        raise MathToolError("invalid_input", "the hypotenuse c must be longer than either leg")

    if len(sides) >= 2:
        sa, sb, sc = sides.get("a"), sides.get("b"), sides.get("c")
        if sc is None:
            sc = sp.sqrt(sa**2 + sb**2)
        elif sa is None:
            sa = sp.sqrt(sc**2 - sb**2)
        elif sb is None:
            sb = sp.sqrt(sc**2 - sa**2)
        elif sp.simplify(sa**2 + sb**2 - sc**2) != 0:
            raise MathToolError("invalid_input", "a^2 + b^2 must equal c^2 for a right triangle")
        angle_a = sp.simplify(sp.atan(sa / sb) * 180 / pi)
    else:
        angle_a = angles["A"] if "A" in angles else 90 - angles["B"]
        rad = angle_a * pi / 180
        (name, value), = sides.items()
        if name == "a":
            sa, sb, sc = value, value / sp.tan(rad), value / sp.sin(rad)
        elif name == "b":
            sa, sb, sc = value * sp.tan(rad), value, value / sp.cos(rad)
        else:
            sa, sb, sc = value * sp.sin(rad), value * sp.cos(rad), value
    sa, sb, sc = (sp.simplify(v) for v in (sa, sb, sc))
    angle_b = sp.simplify(90 - angle_a)
    solved = {"a": sa, "b": sb, "c": sc, "A": angle_a, "B": angle_b, "C": sp.Integer(90)}
    decimals = {k: round(float(sp.N(v)), 4) for k, v in solved.items()}
    text = ", ".join(f"{k} = {v}" for k, v in solved.items())
    return ok(None, result=text, sides={k: solved[k] for k in "abc"}, angles={k: solved[k] for k in "ABC"}, decimals=decimals)


TOOLS = [pythagorean, area, perimeter, volume, surface_area, trig, inverse_trig, solve_right_triangle]
