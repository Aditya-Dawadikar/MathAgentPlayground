"""Word-problem helper tools (issue #9)."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

import sympy as sp

from .core import MathToolError, Number, math_tool, ok, to_number

R = sp.Rational

# Each unit: (dimension exponents, factor to SI base: metre, kilogram, second).
_UNITS: dict[str, tuple[dict[str, int], sp.Rational]] = {}


def _register(names: str, dims: dict[str, int], factor) -> None:
    for name in names.split():
        _UNITS[name] = (dims, R(factor))


L, M, T = {"length": 1}, {"mass": 1}, {"time": 1}
_register("m meter meters metre metres", L, 1)
_register("km kilometer kilometers kilometre kilometres", L, 1000)
_register("cm centimeter centimeters centimetre centimetres", L, "0.01")
_register("mm millimeter millimeters millimetre millimetres", L, "0.001")
_register("mi mile miles", L, "1609.344")
_register("yd yard yards", L, "0.9144")
_register("ft foot feet", L, "0.3048")
_register("in inch inches", L, "0.0254")
_register("kg kilogram kilograms", M, 1)
_register("g gram grams", M, "0.001")
_register("mg milligram milligrams", M, "0.000001")
_register("t tonne tonnes", M, 1000)
_register("lb lbs pound pounds", M, "0.45359237")
_register("oz ounce ounces", M, "0.028349523125")
_register("s sec second seconds", T, 1)
_register("min minute minutes", T, 60)
_register("h hr hour hours", T, 3600)
_register("day days", T, 86400)
_register("week weeks", T, 604800)
_register("year years yr", T, 31536000)  # 365-day year
V = {"length": 3}
_register("l liter liters litre litres", V, "0.001")
_register("ml milliliter milliliters millilitre millilitres", V, "0.000001")
_register("gal gallon gallons", V, "0.003785411784")
_register("qt quart quarts", V, "0.000946352946")
_register("cup cups", V, "0.0002365882365")
A = {"length": 2}
_register("acre acres", A, "4046.8564224")
_register("ha hectare hectares", A, 10000)

_ALIASES = {"mph": "mi/h", "kph": "km/h", "kmh": "km/h"}
_TEMPERATURES = {"c": "C", "celsius": "C", "f": "F", "fahrenheit": "F", "k": "K", "kelvin": "K"}


def _parse_unit(text: str) -> tuple[dict[str, int], sp.Rational]:
    raw = text.strip().lower().replace(" ", "").replace("²", "^2").replace("³", "^3")
    raw = _ALIASES.get(raw, raw)
    raw = re.sub(r"^(sq|square)", "", raw) + ("^2" if re.match(r"^(sq|square)", raw) else "")
    if raw.count("/") > 1:
        raise MathToolError("invalid_input", f"unsupported unit {text!r}")
    numerator, _, denominator = raw.partition("/")
    dims: dict[str, int] = {}
    factor = R(1)
    for side, sign in ((numerator, 1), (denominator, -1)):
        if not side:
            continue
        for piece in re.split(r"[*.]", side):
            match = re.fullmatch(r"([a-z]+)(?:\^?(\d))?", piece)
            if not match or match.group(1) not in _UNITS:
                raise MathToolError("invalid_input", f"unknown unit {piece!r} in {text!r}")
            unit_dims, unit_factor = _UNITS[match.group(1)]
            power = int(match.group(2) or 1) * sign
            factor *= unit_factor**power
            for dim, exp in unit_dims.items():
                dims[dim] = dims.get(dim, 0) + exp * power
    return {k: v for k, v in dims.items() if v}, factor


def _convert_temperature(value: sp.Expr, source: str, target: str) -> sp.Expr:
    celsius = {"C": value, "F": (value - 32) * R(5, 9), "K": value - R("273.15")}[source]
    return {"C": celsius, "F": celsius * R(9, 5) + 32, "K": celsius + R("273.15")}[target]


def _rounded(value: sp.Expr, places: int = 2) -> float:
    """Round half up (money convention), computed from a high-precision value."""
    precise = Decimal(str(sp.N(value, 40)))
    return float(precise.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


@math_tool
def percent(part: Number | None = None, whole: Number | None = None, percent: Number | None = None) -> dict:
    """Percent problems: give any two of part, whole and percent to find the third.

    "What percent is 18 of 72?" -> {"part": 18, "whole": 72} -> 25.
    "What is 15% of 80?" -> {"percent": 15, "whole": 80} -> part 12.
    "30 is 40% of what?" -> {"part": 30, "percent": 40} -> whole 75.
    """
    given = {k: v for k, v in (("part", part), ("whole", whole), ("percent", percent)) if v is not None}
    if len(given) != 2:
        raise MathToolError("invalid_input", "give exactly two of part, whole and percent")
    values = {k: to_number(v, k) for k, v in given.items()}
    if "percent" not in values:
        if values["whole"] == 0:
            raise MathToolError("math_error", "the whole cannot be 0")
        found = values["part"] / values["whole"] * 100
        return ok(found, result=f"{found}%", percent=found, steps=["percent = part / whole * 100"])
    if "part" not in values:
        found = values["percent"] / 100 * values["whole"]
        return ok(found, part=found, steps=["part = percent / 100 * whole"])
    if values["percent"] == 0:
        raise MathToolError("math_error", "the percent cannot be 0 when finding the whole")
    found = values["part"] * 100 / values["percent"]
    return ok(found, whole=found, steps=["whole = part * 100 / percent"])


@math_tool
def percent_change(old: Number, new: Number) -> dict:
    """Percent increase or decrease from old to new.

    Example: {"old": 80, "new": 100} -> 25 (increase); {"old": 50, "new": 40} -> -20 (decrease).
    """
    before, after = to_number(old, "old"), to_number(new, "new")
    if before == 0:
        raise MathToolError("math_error", "percent change from 0 is undefined")
    change = (after - before) / sp.Abs(before) * 100
    direction = "increase" if change > 0 else ("decrease" if change < 0 else "no change")
    return ok(change, result=f"{change}% {direction}", direction=direction, absolute_change=after - before,
              steps=["percent change = (new - old) / |old| * 100"])


@math_tool
def ratio_proportion(a: Number, b: Number, c: Number) -> dict:
    """Solve the proportion a/b = c/x for x (cross-multiply).

    Example: "3 is to 4 as 9 is to x" -> {"a": 3, "b": 4, "c": 9} -> x = 12.
    """
    A, B, C = to_number(a, "a"), to_number(b, "b"), to_number(c, "c")
    if A == 0 or B == 0:
        raise MathToolError("math_error", "a and b must be non-zero")
    x = B * C / A
    return ok(x, result=f"x = {x}", steps=[f"cross-multiply: {A}*x = {B}*{C}", f"x = {B * C}/{A}"])


@math_tool
def unit_convert(value: Number, from_unit: str, to_unit: str) -> dict:
    """Convert between units of length, mass, time, volume, area, speed or temperature.

    Units: m km cm mm mi yd ft in, kg g mg t lb oz, s min h day week year,
    L mL gal qt cup, acre ha, compound units such as km/h, m/s, mph, ft^2, m^3,
    and temperatures C, F, K. Example: {"value": 5, "from_unit": "km", "to_unit": "m"} -> 5000.
    """
    amount = to_number(value, "value")
    source, target = from_unit.strip().lower(), to_unit.strip().lower()
    if source in _TEMPERATURES or target in _TEMPERATURES:
        if source not in _TEMPERATURES or target not in _TEMPERATURES:
            raise MathToolError("invalid_input", f"cannot convert {from_unit} to {to_unit}: incompatible units")
        converted = _convert_temperature(amount, _TEMPERATURES[source], _TEMPERATURES[target])
        return ok(converted, result=f"{_rounded(converted)} {to_unit}")
    source_dims, source_factor = _parse_unit(from_unit)
    target_dims, target_factor = _parse_unit(to_unit)
    if source_dims != target_dims:
        raise MathToolError("invalid_input", f"cannot convert {from_unit} to {to_unit}: incompatible units")
    converted = amount * source_factor / target_factor
    return ok(converted, result=f"{_rounded(converted, 6):g} {to_unit}", factor=source_factor / target_factor)


@math_tool
def simple_compound_interest(principal: Number, rate: Number, time: Number, n: Number | str = 1) -> dict:
    """Simple and compound interest.

    rate is a decimal (0.05 for 5%); values above 1 are treated as percents.
    n is compounding periods per year (1 yearly, 4 quarterly, 12 monthly, 365 daily)
    or "continuous". Example: {"principal": 1000, "rate": 0.05, "time": 3, "n": 12}
    -> compound amount 1161.47, simple interest 150.
    """
    P, r, t = to_number(principal, "principal"), to_number(rate, "rate"), to_number(time, "time")
    steps = []
    if r > 1:
        r = r / 100
        steps.append(f"rate {rate} interpreted as {rate}%")
    if P < 0 or r < 0 or t < 0:
        raise MathToolError("invalid_input", "principal, rate and time must be non-negative")
    simple = P * r * t
    if isinstance(n, str) and n.strip().lower() == "continuous":
        amount = P * sp.exp(r * t)
        steps.append("A = P * e^(r t)")
    else:
        periods = to_number(n, "n")
        if not (periods.is_integer and periods > 0):
            raise MathToolError("invalid_input", "n must be a positive integer or 'continuous'")
        amount = P * (1 + r / periods) ** (periods * t)
        steps.append("A = P * (1 + r/n)^(n t)")
    steps.append("simple interest I = P r t")
    compound_amount = _rounded(amount)
    return ok(None, result=f"compound amount {compound_amount:.2f}, simple interest {_rounded(simple):.2f}",
              steps=steps, compound_amount=compound_amount, compound_interest=_rounded(amount - P),
              simple_interest=_rounded(simple), simple_amount=_rounded(P + simple), exact_compound_amount=amount)


TOOLS = [percent, percent_change, ratio_proportion, unit_convert, simple_compound_interest]
