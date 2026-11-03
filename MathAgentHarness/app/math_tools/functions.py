"""Function tools (issue #6).

Function arguments accept a bare expression ("2x + 1") or a definition such as
"f(x) = 2x + 1" or "y = 2x + 1".
"""

from __future__ import annotations

import re

import sympy as sp
from sympy.calculus.util import continuous_domain, function_range

from .core import MathToolError, Number, ensure_defined, math_tool, ok, parse_expression, parse_variable, resolve_variable, to_number
from .equations import _interval_text, _relational_text

_DEFINITION = re.compile(r"^\s*(?:[A-Za-z]\s*\(\s*([A-Za-z])\s*\)|y)\s*=\s*")


def _function(text: str, variable: str | None, label: str = "function") -> tuple[sp.Expr, sp.Symbol]:
    """Parse a function, honouring the variable named in an "f(t) = ..." definition."""
    if not isinstance(text, str):
        expr = parse_expression(text, label)
        return expr, resolve_variable(expr, variable) if expr.free_symbols else parse_variable(variable or "x")
    match = _DEFINITION.match(text)
    declared = None
    if match:
        declared = match.group(1)
        text = text[match.end():]
    expr = parse_expression(text, label)
    name = variable or declared
    if name:
        return expr, parse_variable(name)
    if not expr.free_symbols:
        return expr, sp.Symbol("x")
    return expr, resolve_variable(expr, None)


def _real(expr: sp.Expr, var: sp.Symbol) -> tuple[sp.Expr, sp.Symbol]:
    real_var = sp.Symbol(var.name, real=True)
    return expr.subs(var, real_var), real_var


@math_tool
def evaluate_function(expression: str, value: Number, variable: str | None = None) -> dict:
    """Evaluate a function at a value, e.g. f(4).

    Example: {"expression": "x^2 - 3x", "value": 4} -> 4;
    {"expression": "f(x) = 2x + 1", "value": "1/2"} -> 2.
    """
    expr, var = _function(expression, variable)
    point = to_number(value, "value")
    result = sp.simplify(expr.subs(var, point))
    if result.has(sp.zoo, sp.nan):
        raise MathToolError("math_error", f"{var} = {point} is not in the domain of the function")
    if not sp.N(result).is_real:
        raise MathToolError("math_error", f"the function is not real-valued at {var} = {point}")
    return ok(ensure_defined(result), steps=[f"substitute {var} = {point}"])


@math_tool
def compose_functions(f: str, g: str, variable: str = "x") -> dict:
    """Composition f(g(x)): substitute g into f.

    Example: {"f": "2x + 1", "g": "x^2"} -> "2*x**2 + 1". For g(f(x)), swap the arguments.
    """
    outer, var = _function(f, variable, "f")
    inner, _ = _function(g, variable, "g")
    composed = sp.simplify(outer.subs(var, inner))
    if composed.is_polynomial(var):
        composed = sp.expand(composed)
    return ok(composed, steps=[f"replace {var} in f with ({inner})"])


@math_tool
def inverse_function(expression: str, variable: str | None = None) -> dict:
    """Inverse function f^-1(x): swap x and y, then solve for y.

    If f is not one-to-one (e.g. a parabola) every branch is returned with
    is_one_to_one false and a suggested domain restriction.
    Example: {"expression": "3x - 4"} -> "x/3 + 4/3"; {"expression": "x^3 + 1"} -> "(x - 1)**(1/3)".
    """
    expr, var = _function(expression, variable)
    if var not in expr.free_symbols:
        raise MathToolError("invalid_input", "a constant function has no inverse")
    y = sp.Symbol("_y", real=True)
    real_var = sp.Symbol(var.name, real=True)
    solutions = sp.solve(sp.Eq(y, expr.subs(var, real_var)), real_var)
    # Drop complex branches (e.g. the non-real cube roots of x^3 + 1).
    branches = [sp.simplify(b.subs(y, var)) for b in solutions if not b.has(sp.I)]
    if not branches:
        raise MathToolError("unsupported", "could not find an inverse for this function")
    steps = [f"write y = {expr}", f"swap {var} and y, then solve for y"]
    if len(branches) == 1:
        inverse = branches[0]
        check = sp.simplify(expr.subs(var, inverse) - var)
        steps.append(f"check: f(f^-1({var})) = {var}" + ("" if check == 0 else " (holds on the restricted domain)"))
        return ok(inverse, steps=steps, is_one_to_one=True, branches=branches)
    restriction = None
    poly = sp.Poly(expr, var) if expr.is_polynomial(var) else None
    if poly is not None and poly.degree() == 2:
        a, b, _ = poly.all_coeffs()
        restriction = f"{var} >= {sp.simplify(-b / (2 * a))}"
    return ok(None, result=" or ".join(map(str, branches)), steps=steps + ["the function is not one-to-one"],
              is_one_to_one=False, branches=branches, suggested_domain_restriction=restriction)


@math_tool
def domain_range(expression: str, variable: str | None = None) -> dict:
    """Domain and range of a real function, in interval and inequality notation.

    Handles denominators, even roots and logarithms.
    Example: {"expression": "1/(x - 2)"} -> domain "(-inf, 2) U (2, inf)", range "(-inf, 0) U (0, inf)";
    {"expression": "sqrt(x - 3)"} -> domain "[3, inf)", range "[0, inf)".
    """
    expr, var = _function(expression, variable)
    expr, var = _real(expr, var)
    domain = continuous_domain(expr, var, sp.S.Reals)
    steps = []
    try:
        range_ = function_range(expr, var, domain)
        range_text = _interval_text(range_)
    except Exception:  # SymPy cannot always determine the range
        range_, range_text = None, None
        steps.append("the range could not be determined exactly")
    return ok(None, result=f"domain {_interval_text(domain)}, range {range_text}", steps=steps,
              domain=_interval_text(domain), domain_inequality=_relational_text(domain, var) if domain != sp.S.Reals else "all real numbers",
              range=range_text, variable=var)


@math_tool
def find_intercepts(expression: str, variable: str | None = None) -> dict:
    """x-intercepts (zeros) and y-intercept of a function as (x, y) points.

    Example: {"expression": "x^2 - 4"} -> x-intercepts [[-2, 0], [2, 0]], y-intercept [0, -4].
    """
    expr, var = _function(expression, variable)
    expr, var = _real(expr, var)
    zeros = sp.solveset(expr, var, domain=continuous_domain(expr, var, sp.S.Reals))
    steps = []
    if isinstance(zeros, sp.FiniteSet):
        x_intercepts = [[z, 0] for z in sorted(zeros, key=lambda v: float(sp.N(v)))]
    elif zeros == sp.EmptySet:
        x_intercepts = []
    else:
        x_intercepts = []
        steps.append(f"infinitely many or non-isolated zeros: {zeros}")
    y_value = expr.subs(var, 0)
    y_intercept = None if y_value.has(sp.zoo, sp.nan) or not sp.N(y_value).is_real else [0, sp.simplify(y_value)]
    if y_intercept is None:
        steps.append(f"{var} = 0 is not in the domain, so there is no y-intercept")
    return ok(None, result=f"x-intercepts {[[str(a), str(b)] for a, b in x_intercepts]}, y-intercept {y_intercept}",
              steps=steps, x_intercepts=x_intercepts, y_intercept=y_intercept)


@math_tool
def vertex_of_parabola(a: Number, b: Number, c: Number) -> dict:
    """Vertex, axis of symmetry, direction and min/max of y = a*x^2 + b*x + c.

    Example: {"a": 1, "b": -4, "c": 3} -> vertex [2, -1], axis "x = 2", opens "up", minimum -1.
    """
    A, B, C = to_number(a, "a"), to_number(b, "b"), to_number(c, "c")
    if A == 0:
        raise MathToolError("invalid_input", "a must not be 0 for a parabola")
    h = sp.simplify(-B / (2 * A))
    k = sp.simplify(A * h**2 + B * h + C)
    opens = "up" if A.is_positive else "down"
    extreme = "minimum" if opens == "up" else "maximum"
    return ok(None, result=f"vertex ({h}, {k}), axis of symmetry x = {h}, opens {opens}",
              steps=[f"h = -b/(2a) = {h}", f"k = f(h) = {k}"], vertex=[h, k], axis_of_symmetry=f"x = {h}",
              opens=opens, extreme_type=extreme, extreme_value=k)


TOOLS = [evaluate_function, compose_functions, inverse_function, domain_range, find_intercepts, vertex_of_parabola]
