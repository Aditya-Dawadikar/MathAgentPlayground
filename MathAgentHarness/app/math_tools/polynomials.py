"""Polynomial tools (issue #5)."""

from __future__ import annotations

import sympy as sp

from .core import MathToolError, Number, math_tool, ok, parse_expression, resolve_variable, to_number

_DEGREE_NAMES = {0: "constant", 1: "linear", 2: "quadratic", 3: "cubic", 4: "quartic", 5: "quintic"}
_TERM_NAMES = {1: "monomial", 2: "binomial", 3: "trinomial"}


def _poly(expression: str, variable: str | None, label: str = "expression") -> tuple[sp.Poly, sp.Symbol]:
    expr = parse_expression(expression, label)
    var = resolve_variable(expr, variable)
    try:
        poly = sp.Poly(sp.expand(expr), var)
    except sp.PolynomialError as exc:
        raise MathToolError("invalid_input", f"{label} is not a polynomial in {var}: {expression!r}") from exc
    if any(not c.is_number for c in poly.all_coeffs()) or poly.is_zero is None:
        raise MathToolError("invalid_input", f"{label} must be a polynomial in {var} with numeric coefficients")
    for atom in sp.expand(expr).atoms(sp.Pow):
        if atom.base.has(var) and not (atom.exp.is_integer and atom.exp >= 0):
            raise MathToolError("invalid_input", f"{label} is not a polynomial in {var}: {expression!r}")
    return poly, var


@math_tool
def polynomial_degree(expression: str, variable: str | None = None) -> dict:
    """Degree, leading coefficient, coefficients and name of a polynomial.

    Example: {"expression": "4x^3 - x + 7"} -> degree 3, leading coefficient 4,
    name "cubic trinomial", standard form "4*x**3 - x + 7".
    """
    poly, var = _poly(expression, variable)
    degree = poly.degree()
    terms = len(poly.terms())
    name = " ".join(filter(None, [_DEGREE_NAMES.get(degree, f"degree {degree}"), _TERM_NAMES.get(terms, "polynomial")]))
    return ok(sp.Integer(degree), degree=degree, leading_coefficient=poly.LC(), coefficients=poly.all_coeffs(),
              constant_term=poly.all_coeffs()[-1], standard_form=poly.as_expr(), name=name, variable=var)


@math_tool
def polynomial_divide(dividend: str, divisor: str, variable: str | None = None) -> dict:
    """Divide polynomials, returning quotient and remainder.

    Uses synthetic division when the divisor is x - k (and reports p(k) = remainder,
    the remainder theorem), otherwise long division.
    Example: {"dividend": "x^3 - 1", "divisor": "x - 1"} -> quotient "x**2 + x + 1", remainder 0.
    """
    top = parse_expression(dividend, "dividend")
    bottom = parse_expression(divisor, "divisor")
    var = resolve_variable(top + bottom, variable)
    p, _ = _poly(dividend, str(var), "dividend")
    d, _ = _poly(divisor, str(var), "divisor")
    if d.is_zero:
        raise MathToolError("math_error", "cannot divide by the zero polynomial")
    quotient, remainder = sp.div(p, d)
    steps = []
    if d.degree() == 1 and d.LC() == 1:
        k = -d.all_coeffs()[-1]
        method = "synthetic division"
        steps.append(f"synthetic division by {k} using coefficients {p.all_coeffs()}")
        steps.append(f"remainder theorem: p({k}) = {p.eval(k)}")
    else:
        method = "long division"
    q_expr, r_expr = quotient.as_expr(), remainder.as_expr()
    result = f"{q_expr}" + (f" remainder {r_expr}" if r_expr != 0 else "")
    return ok(q_expr, result=result, steps=steps, quotient=q_expr, remainder=r_expr, method=method,
              divides_evenly=r_expr == 0, as_mixed=sp.Add(q_expr, r_expr / d.as_expr(), evaluate=False))


@math_tool
def find_roots(polynomial: str, variable: str | None = None) -> dict:
    """All real and complex roots of a polynomial with multiplicities.

    Exact where possible; falls back to numeric roots for unsolvable degrees.
    Example: {"polynomial": "x^3 - 6x^2 + 11x - 6"} -> real roots [1, 2, 3].
    """
    poly, var = _poly(polynomial, variable, "polynomial")
    if poly.degree() < 1:
        raise MathToolError("invalid_input", "a constant polynomial has no roots to find")
    exact = sp.roots(poly)
    steps = []
    if sum(exact.values()) < poly.degree():
        steps.append("no exact radical roots found; using numeric approximations")
        exact = {}
        for r in poly.nroots():
            exact[r] = exact.get(r, 0) + 1
    entries = [{"value": r, "multiplicity": m, "is_real": bool(r.is_real)} for r, m in exact.items()]
    real = sorted([e for e in entries if e["is_real"]], key=lambda e: float(sp.N(e["value"])))
    complex_ = sorted([e for e in entries if not e["is_real"]], key=lambda e: sp.default_sort_key(e["value"]))
    real_values = [e["value"] for e in real]
    text = ", ".join(f"{var} = {e['value']}" + (f" (multiplicity {e['multiplicity']})" if e["multiplicity"] > 1 else "") for e in real + complex_)
    envelope = ok(None, result=text, steps=steps, roots=real + complex_, real_roots=real_values,
                  complex_roots=[e["value"] for e in complex_], factored=sp.factor(poly.as_expr()))
    envelope["exact"] = [str(v) for v in real_values]
    return envelope


@math_tool
def complete_the_square(expression: str, variable: str | None = None) -> dict:
    """Rewrite a quadratic a*x^2 + b*x + c as a*(x - h)^2 + k (vertex form).

    Works for any leading coefficient. Example: {"expression": "x^2 + 6x + 5"} ->
    "(x + 3)**2 - 4" with h = -3, k = -4; {"expression": "2x^2 + 8x + 3"} -> "2*(x + 2)**2 - 5".
    """
    poly, var = _poly(expression, variable)
    if poly.degree() != 2:
        raise MathToolError("invalid_input", f"completing the square needs a quadratic, got degree {poly.degree()}")
    a, b, c = poly.all_coeffs()
    h = sp.simplify(-b / (2 * a))
    k = sp.simplify(c - b**2 / (4 * a))
    vertex_form = a * (var - h) ** 2 + k
    steps = [
        f"factor {a} from the x terms" if a != 1 else "the leading coefficient is 1",
        f"half of {sp.simplify(b / a)} is {sp.simplify(b / (2 * a))}; add and subtract its square {sp.simplify((b / (2 * a)) ** 2)}",
        f"vertex form: {vertex_form}",
    ]
    return ok(vertex_form, steps=steps, a=a, h=h, k=k, vertex=[h, k])


@math_tool
def discriminant(a: Number, b: Number, c: Number) -> dict:
    """Discriminant b^2 - 4ac of a*x^2 + b*x + c and what it says about the roots.

    Example: {"a": 1, "b": 2, "c": 5} -> -16, nature "two complex conjugate roots", real_root_count 0.
    """
    A, B, C = to_number(a, "a"), to_number(b, "b"), to_number(c, "c")
    if A == 0:
        raise MathToolError("invalid_input", "a must not be 0 for a quadratic")
    value = sp.simplify(B**2 - 4 * A * C)
    if value.is_positive:
        nature, count = "two distinct real roots", 2
    elif value.is_zero:
        nature, count = "one repeated real root", 1
    else:
        nature, count = "two complex conjugate roots", 0
    return ok(value, steps=[f"({B})^2 - 4({A})({C}) = {value}"], nature=nature, real_root_count=count,
              rational_roots=bool(value.is_nonnegative and sp.sqrt(value).is_rational))


TOOLS = [polynomial_degree, polynomial_divide, find_roots, complete_the_square, discriminant]
