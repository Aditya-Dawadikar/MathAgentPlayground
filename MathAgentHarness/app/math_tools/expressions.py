"""Expression manipulation tools (issue #3)."""

from __future__ import annotations

import sympy as sp

from .core import MathToolError, ensure_defined, math_tool, ok, parse_expression, parse_variable


def _positive(expr: sp.Expr) -> tuple[sp.Expr, dict]:
    """Replace symbols with positive ones (school convention for radicals)."""
    forward = {s: sp.Symbol(s.name, positive=True) for s in expr.free_symbols}
    backward = {v: k for k, v in forward.items()}
    return expr.subs(forward), backward


def restrictions(expr: sp.Expr) -> list[str]:
    """Values excluded because a denominator would be zero, e.g. ["x != 1"]."""
    denominator = sp.factor(sp.denom(sp.together(expr)))
    if denominator.is_number:
        return []
    symbols = sorted(denominator.free_symbols, key=lambda s: s.name)
    if len(symbols) == 1:
        var = symbols[0]
        zeros = sp.solveset(denominator, var, domain=sp.S.Reals)
        if isinstance(zeros, sp.FiniteSet):
            return [f"{var} != {z}" for z in sorted(zeros, key=sp.default_sort_key)]
    return [f"{denominator} != 0"]


@math_tool
def simplify(expression: str) -> dict:
    """Simplify an algebraic expression: combine like terms and cancel common factors.

    Also reports excluded values for rational expressions.
    Example: {"expression": "3x + 2 - x + 5"} -> "2*x + 7";
    {"expression": "(x^2 - 1)/(x - 1)"} -> "x + 1" with restrictions ["x != 1"].
    """
    expr = parse_expression(expression)
    excluded = restrictions(expr)
    simplified = sp.simplify(expr)
    if simplified.is_polynomial() and not simplified.is_number:
        simplified = sp.expand(simplified)
    return ok(ensure_defined(simplified), restrictions=excluded)


@math_tool
def expand(expression: str) -> dict:
    """Expand products and powers (distribute / FOIL).

    Example: {"expression": "(x+2)(x-3)"} -> "x**2 - x - 6".
    """
    expr = parse_expression(expression)
    return ok(sp.expand(expr))


def _perfect_root(term: sp.Expr, n: int) -> sp.Expr | None:
    positive_term, backward = _positive(sp.Abs(term))
    root = sp.root(positive_term, n)
    root = sp.powsimp(sp.simplify(root), force=True)
    symbols = list(root.free_symbols)
    if root.has(sp.Pow) and any(p.exp.is_Rational and not p.exp.is_integer for p in root.atoms(sp.Pow)):
        return None
    if symbols and not root.is_polynomial(*symbols):
        return None
    coefficient = root.as_coeff_Mul()[0]
    if not coefficient.is_Rational:
        return None
    return root.subs(backward)


def _factor_techniques(expr: sp.Expr, factored: sp.Expr) -> list[str]:
    expanded = sp.expand(expr)
    terms = sp.Add.make_args(expanded)
    techniques = []
    gcf = sp.Integer(0)
    for term in terms:
        gcf = sp.gcd(gcf, term)
    gcf = sp.Abs(gcf) if gcf.is_number else gcf
    if gcf != 1 and len(terms) > 1:
        techniques.append(f"greatest common factor {gcf}")
        rest = sp.expand(sp.cancel(expanded / gcf))
    else:
        rest = expanded
    rest_terms = sp.Add.make_args(rest)
    if len(rest_terms) == 2:
        first, second = rest_terms
        if (first.could_extract_minus_sign()) != (second.could_extract_minus_sign()):
            if _perfect_root(first, 2) is not None and _perfect_root(second, 2) is not None:
                techniques.append("difference of squares")
        if _perfect_root(first, 3) is not None and _perfect_root(second, 3) is not None:
            same_sign = first.could_extract_minus_sign() == second.could_extract_minus_sign()
            techniques.append("sum of cubes" if same_sign else "difference of cubes")
    elif len(rest_terms) == 3:
        squared = any(isinstance(f, sp.Pow) and f.exp == 2 for f in sp.Mul.make_args(factored))
        techniques.append("perfect square trinomial" if squared else "trinomial factoring")
    elif len(rest_terms) == 4:
        techniques.append("factoring by grouping")
    return techniques


@math_tool
def factor(expression: str) -> dict:
    """Factor a polynomial completely over the rationals.

    Handles greatest common factor, difference of squares, trinomials,
    sum/difference of cubes, and grouping; the technique is listed in steps.
    Example: {"expression": "x^2 - 9"} -> "(x - 3)*(x + 3)";
    {"expression": "6x^2 + 9x"} -> "3*x*(2*x + 3)".
    """
    expr = parse_expression(expression)
    factored = sp.factor(expr)
    factorable = sp.Mul.make_args(factored) != sp.Mul.make_args(sp.expand(expr)) or isinstance(factored, sp.Pow)
    if not factorable:
        return ok(factored, steps=["no factorization over the rationals (prime polynomial)"], is_factorable=False)
    return ok(factored, steps=_factor_techniques(expr, factored), is_factorable=True)


@math_tool
def substitute(expression: str, values: dict[str, str | int | float]) -> dict:
    """Substitute values (numbers or expressions) for variables and simplify.

    Example: {"expression": "x^2 + 2y", "values": {"x": 3, "y": 1}} -> 11.
    """
    expr = parse_expression(expression)
    if not values:
        raise MathToolError("invalid_input", "values must map at least one variable to a value")
    mapping = {parse_variable(name): parse_expression(val, f"value for {name}") for name, val in values.items()}
    unknown = [s.name for s in mapping if s not in expr.free_symbols]
    if unknown:
        raise MathToolError("invalid_input", f"variable(s) {', '.join(unknown)} do not appear in the expression")
    result = ensure_defined(sp.simplify(expr.subs(mapping)), "substituted expression")
    return ok(result)


@math_tool
def rationalize_denominator(expression: str) -> dict:
    """Remove radicals from a denominator.

    Example: {"expression": "1/sqrt(2)"} -> "sqrt(2)/2";
    {"expression": "1/(1 + sqrt(3))"} -> "(-1 + sqrt(3))/2".
    """
    expr = parse_expression(expression)
    denominator = sp.denom(sp.together(expr))
    if not any(isinstance(p, sp.Pow) and p.exp.is_Rational and not p.exp.is_integer for p in denominator.atoms(sp.Pow)):
        return ok(expr, steps=["the denominator has no radicals"])
    result = sp.radsimp(expr)
    steps = ["multiply numerator and denominator by the conjugate" if len(sp.Add.make_args(denominator)) > 1 else "multiply numerator and denominator by the radical"]
    return ok(result, steps=steps)


def _radical_form(expr: sp.Expr) -> sp.Expr:
    """Rewrite x**(3/2) as x*sqrt(x) and merge radicals with the same index (school form)."""
    if not isinstance(expr, sp.Mul) and not isinstance(expr, sp.Pow):
        return expr
    outside, inside = [], {}
    for factor_ in sp.Mul.make_args(expr):
        base, exponent = factor_.as_base_exp()
        if exponent.is_Rational and not exponent.is_integer:
            whole, remainder = divmod(exponent.p, exponent.q)
            if whole:
                outside.append(base**whole)
            inside.setdefault(exponent.q, []).append(base**remainder)
        else:
            outside.append(factor_)
    radicals = [
        sp.sqrt(sp.Mul(*bases), evaluate=False) if index == 2 else sp.root(sp.Mul(*bases), index, evaluate=False)
        for index, bases in inside.items()
    ]
    return sp.Mul(*outside, *radicals, evaluate=False)


@math_tool
def simplify_radical(expression: str) -> dict:
    """Simplify radicals, pulling perfect powers out of the root.

    Variables are assumed non-negative, as in school algebra.
    Example: {"expression": "sqrt(72)"} -> "6*sqrt(2)"; {"expression": "sqrt(50x^3)"} -> "5*x*sqrt(2*x)".
    """
    expr = parse_expression(expression)
    positive, backward = _positive(expr)
    simplified = sp.simplify(sp.powsimp(sp.radsimp(positive), force=True)).subs(backward)
    return ok(_radical_form(simplified), steps=["variables assumed non-negative"] if expr.free_symbols else [])


@math_tool
def combine_rational(expression: str) -> dict:
    """Add or subtract algebraic fractions into a single simplified fraction.

    Reports excluded values where any denominator is zero.
    Example: {"expression": "1/x + 2/(x+1)"} -> "(3*x + 1)/(x*(x + 1))" with restrictions ["x != -1", "x != 0"].
    """
    expr = parse_expression(expression)
    excluded = restrictions(expr)
    combined = sp.factor(sp.cancel(sp.together(expr)))
    numerator, denominator = sp.fraction(combined)
    return ok(combined, restrictions=excluded, numerator=numerator, denominator=denominator)


TOOLS = [simplify, expand, factor, substitute, rationalize_denominator, simplify_radical, combine_rational]
