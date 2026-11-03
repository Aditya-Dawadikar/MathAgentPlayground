"""Equation, system and inequality solvers (issue #4).

Every solver verifies candidate solutions against the *original* equation, so
extraneous roots (from squaring radicals or clearing denominators) are reported
separately instead of being returned as answers.

Result fields:
    solution_type: "finite" | "none" | "all_real" | "infinite" (systems)
    solutions:     list of exact values (or dicts for systems)
"""

from __future__ import annotations

import sympy as sp

from .core import (
    MathToolError,
    Number,
    math_tool,
    ok,
    parse_equation,
    parse_expression,
    parse_relation,
    parse_variable,
    resolve_variable,
    to_number,
)
from .expressions import restrictions

_TOLERANCE = 1e-9


def _real(symbol: sp.Symbol) -> sp.Symbol:
    return sp.Symbol(symbol.name, real=True)


def _as_real(expr, variables):
    """Swap the given symbols for real-valued ones (school algebra works over the reals).

    Relations are rebuilt unevaluated so e.g. ``x^2 + 1 = 0`` is not collapsed to ``False``.
    """
    mapping = {v: _real(v) for v in variables}
    return _subs(expr, mapping), [mapping[v] for v in variables]


def _subs(expr, mapping):
    if isinstance(expr, sp.Equality):
        return sp.Eq(expr.lhs.subs(mapping), expr.rhs.subs(mapping), evaluate=False)
    if isinstance(expr, sp.core.relational.Relational):
        return sp.Rel(expr.lhs.subs(mapping), expr.rhs.subs(mapping), expr.rel_op, evaluate=False)
    if isinstance(expr, sp.And):
        return sp.And(*(_subs(a, mapping) for a in expr.args), evaluate=False)
    return expr.subs(mapping)


def _relation_symbols(relations) -> set:
    """Symbols on either side, even if they cancel (``x + 1 = x + 2``)."""
    return set().union(*(r.lhs.free_symbols | r.rhs.free_symbols for r in relations))


def _relation_holds(relation, values: dict) -> bool | None:
    """True/False if the relation holds at ``values``; None if it is undefined there."""
    lhs = relation.lhs.subs(values)
    rhs = relation.rhs.subs(values)
    if lhs.has(sp.zoo, sp.nan) or rhs.has(sp.zoo, sp.nan):
        return None
    if isinstance(relation, sp.Equality):
        diff = sp.simplify(lhs - rhs)
        if diff.has(sp.zoo, sp.nan):
            return None
        if diff == 0:
            return True
        if diff.free_symbols:
            return False
        return abs(complex(sp.N(diff))) < _TOLERANCE
    lhs_value, rhs_value = sp.N(lhs), sp.N(rhs)
    if not (lhs_value.is_real and rhs_value.is_real):
        return None
    return bool(relation.func(lhs_value, rhs_value))


def _clear_denominators(eq: sp.Equality) -> sp.Expr:
    """Multiply both sides by the LCD, like a student would.

    Unlike ``together``/``cancel`` this keeps candidates such as x = 1 in
    ``x/(x-1) = 1/(x-1)``, which verification then rejects as extraneous.
    """
    terms = sp.Add.make_args(eq.lhs) + sp.Add.make_args(eq.rhs)
    denominators = [sp.denom(term) for term in terms]
    try:
        lcd = sp.lcm_list(denominators)
    except Exception:  # non-polynomial denominators
        lcd = sp.Mul(*denominators)
    return sp.expand(eq.lhs * lcd - eq.rhs * lcd)


def _format_solutions(var: sp.Symbol, solutions: list) -> str:
    return " or ".join(f"{var} = {s}" for s in solutions)


@math_tool
def solve_equation(equation: str, variable: str | None = None) -> dict:
    """Solve a single equation for one variable over the real numbers.

    Handles linear, quadratic, polynomial, rational, radical, absolute-value and
    exponential equations. Candidate roots are checked against the original
    equation; ones that fail are returned in "extraneous".
    Example: {"equation": "2x + 3 = 11"} -> "x = 4";
    {"equation": "sqrt(x + 2) = x"} -> "x = 2" with extraneous ["-1"].
    """
    eq = parse_equation(equation)
    symbols = _relation_symbols([eq])
    var = resolve_variable(sp.Add(*symbols) if symbols else sp.Integer(0), variable)
    if var not in symbols:
        raise MathToolError("invalid_input", f"variable {var} does not appear in the equation")
    eq, (var,) = _as_real(eq, [var])
    expr = eq.lhs - eq.rhs
    excluded = restrictions(expr)

    numerator = _clear_denominators(eq)
    if sp.simplify(numerator) == 0:
        return ok(None, result="all real numbers" + (f" except {', '.join(excluded)}" if excluded else ""),
                  solution_type="all_real", solutions=[], restrictions=excluded, variable=var)
    if numerator.is_number:
        return ok(None, result="no solution", solution_type="none", solutions=[],
                  steps=[f"the equation reduces to {numerator} = 0, which is false"], variable=var)

    candidates = sp.solve(numerator, var, check=False)
    valid, extraneous, complex_roots = [], [], []
    for candidate in candidates:
        if candidate.free_symbols - {var}:
            raise MathToolError("invalid_input", "the equation has other variables; use solve_for instead")
        if not sp.N(candidate).is_real:
            complex_roots.append(candidate)
            continue
        holds = _relation_holds(eq, {var: candidate})
        (valid if holds else extraneous).append(sp.nsimplify(candidate) if candidate.is_Float else candidate)
    valid = sorted(set(valid), key=lambda v: float(sp.N(v)))
    extraneous = sorted(set(extraneous), key=sp.default_sort_key)
    steps = [f"checked {len(candidates)} candidate solution(s) in the original equation"]
    if extraneous:
        steps.append(f"rejected extraneous solution(s): {', '.join(map(str, extraneous))}")
    if not valid:
        return ok(None, result="no real solution", solution_type="none", solutions=[], extraneous=extraneous,
                  complex_solutions=complex_roots, restrictions=excluded, steps=steps, variable=var)
    value = valid[0] if len(valid) == 1 else None
    envelope = ok(value, result=_format_solutions(var, valid), steps=steps, solution_type="finite", solutions=valid,
                  positive_solutions=[v for v in valid if v.is_positive],
                  extraneous=extraneous, complex_solutions=complex_roots, restrictions=excluded, variable=var)
    if value is None:
        envelope["exact"] = [str(v) for v in valid]
        envelope["decimal"] = [float(sp.N(v)) for v in valid]
    return envelope


@math_tool
def solve_quadratic(a: Number, b: Number, c: Number) -> dict:
    """Solve a*x^2 + b*x + c = 0 with the quadratic formula.

    Returns exact roots (irrational and complex included), the discriminant and
    the nature of the roots. Example: {"a": 1, "b": -5, "c": 6} -> roots [2, 3],
    discriminant 1, nature "two distinct real roots".
    """
    A, B, C = to_number(a, "a"), to_number(b, "b"), to_number(c, "c")
    if A == 0:
        raise MathToolError("invalid_input", "a must not be 0 (that is a linear equation; use solve_equation)")
    discriminant = sp.simplify(B**2 - 4 * A * C)
    sqrt_d = sp.sqrt(discriminant)
    roots = [sp.simplify((-B - sqrt_d) / (2 * A)), sp.simplify((-B + sqrt_d) / (2 * A))]
    if discriminant.is_positive:
        nature = "two distinct real roots" + (" (rational)" if sqrt_d.is_rational else " (irrational)")
        roots = sorted(roots, key=lambda r: float(sp.N(r)))
    elif discriminant.is_zero:
        nature = "one repeated real root"
        roots = roots[:1]
    else:
        nature = "two complex conjugate roots (no real roots)"
    steps = [
        "x = (-b ± sqrt(b^2 - 4ac)) / (2a)",
        f"discriminant = ({B})^2 - 4({A})({C}) = {discriminant}",
    ]
    x = sp.Symbol("x")
    real_roots = [r for r in roots if r.is_real]
    # "solutions" mirrors solve_equation (real solutions) so plans can use $step_N.solutions[i] with either tool.
    envelope = ok(None, result=_format_solutions(x, roots), steps=steps, roots=roots, discriminant=discriminant,
                  nature=nature, real_roots=real_roots, solutions=real_roots,
                  positive_solutions=[r for r in real_roots if r.is_positive],
                  solution_type="finite" if real_roots else "none")
    envelope["exact"] = [str(r) for r in roots]
    envelope["decimal"] = [float(sp.N(r)) if r.is_real else None for r in roots]
    return envelope


def _is_linear(expr: sp.Expr, variables: list[sp.Symbol]) -> bool:
    try:
        return sp.Poly(expr, *variables).total_degree() <= 1
    except sp.PolynomialError:
        return False


@math_tool
def solve_system(equations: list[str], variables: list[str] | None = None) -> dict:
    """Solve a system of equations (linear 2x2/3x3 or simple nonlinear).

    Reports inconsistent systems (solution_type "none") and dependent systems
    with infinitely many solutions (solution_type "infinite", parametric form).
    Example: {"equations": ["x + y = 10", "x - y = 2"]} -> "x = 6, y = 4".
    """
    if not equations or len(equations) < 2:
        raise MathToolError("invalid_input", "provide at least two equations")
    parsed = [parse_equation(e, f"equation {i + 1}") for i, e in enumerate(equations)]
    exprs = [eq.lhs - eq.rhs for eq in parsed]
    free = set().union(*(e.free_symbols for e in exprs))
    if variables:
        syms = [parse_variable(v) for v in variables]
    else:
        syms = sorted(free, key=lambda s: s.name)
    if not syms:
        raise MathToolError("invalid_input", "the system has no variables")
    exprs, syms = _as_real(sp.Matrix(exprs), syms)
    exprs = list(exprs)
    real_parsed = [sp.Eq(e, 0) for e in exprs]

    if all(_is_linear(e, syms) for e in exprs):
        method = "linear elimination"
        result_set = sp.linsolve(exprs, syms)
        if result_set == sp.EmptySet:
            return ok(None, result="no solution (inconsistent system)", solution_type="none", solutions=[], method=method)
        (solution,) = result_set
        params = set().union(*(s.free_symbols for s in solution)) & set(syms)
        mapping = dict(zip(syms, solution))
        if params:
            text = ", ".join(f"{k} = {v}" for k, v in mapping.items())
            return ok(None, result=f"infinitely many solutions: {text}", solution_type="infinite",
                      solutions=[mapping], free_variables=sorted(params, key=lambda s: s.name), method=method)
        solutions = [mapping]
    else:
        method = "substitution (nonlinear)"
        candidates = sp.solve(exprs, syms, dict=True)
        solutions = [
            s for s in candidates
            if all(sp.N(v).is_real for v in s.values()) and all(_relation_holds(eq, s) for eq in real_parsed)
        ]
        if not solutions:
            return ok(None, result="no real solution", solution_type="none", solutions=[], method=method)

    texts = [", ".join(f"{k} = {v}" for k, v in s.items()) for s in solutions]
    envelope = ok(None, result=" or ".join(f"({t})" if len(solutions) > 1 else t for t in texts),
                  solution_type="finite", solutions=solutions, method=method)
    envelope["exact"] = envelope["solutions"]
    return envelope


def _interval_text(solution_set) -> str:
    def bound(value):
        if value == sp.oo:
            return "inf"
        if value == -sp.oo:
            return "-inf"
        return str(value)

    if solution_set == sp.EmptySet:
        return "no solution"
    if solution_set == sp.S.Reals:
        return "(-inf, inf)"
    if isinstance(solution_set, sp.Interval):
        left = "(" if solution_set.left_open else "["
        right = ")" if solution_set.right_open else "]"
        return f"{left}{bound(solution_set.start)}, {bound(solution_set.end)}{right}"
    if isinstance(solution_set, sp.FiniteSet):
        return "{" + ", ".join(str(v) for v in solution_set) + "}"
    if isinstance(solution_set, sp.Union):
        return " U ".join(_interval_text(part) for part in solution_set.args)
    if isinstance(solution_set, sp.Complement):
        return f"{_interval_text(solution_set.args[0])} excluding {solution_set.args[1]}"
    return str(solution_set)


def _relational_text(solution_set, var) -> str:
    """School notation: ``-1 < x <= 2``, ``x <= -1 or x >= 3``."""
    if isinstance(solution_set, sp.Union):
        return " or ".join(_relational_text(part, var) for part in solution_set.args)
    if isinstance(solution_set, sp.Interval):
        start, end = solution_set.start, solution_set.end
        left = "<" if solution_set.left_open else "<="
        right = "<" if solution_set.right_open else "<="
        if start == -sp.oo:
            return f"{var} {right} {end}"
        if end == sp.oo:
            return f"{var} {'>' if solution_set.left_open else '>='} {start}"
        return f"{start} {left} {var} {right} {end}"
    if isinstance(solution_set, sp.FiniteSet):
        return " or ".join(f"{var} = {v}" for v in solution_set)
    return str(solution_set.as_relational(var))


@math_tool
def solve_inequality(inequality: str, variable: str | None = None) -> dict:
    """Solve an inequality over the real numbers, including compound and absolute-value ones.

    Returns inequality notation and interval notation. Remember the sign flips
    when dividing by a negative number; the steps say when that happens.
    Example: {"inequality": "3x - 2 <= 10"} -> "x <= 4", interval "(-inf, 4]";
    {"inequality": "-2x > 6"} -> "x < -3"; {"inequality": "1 < 2x + 3 <= 7"} -> "-1 < x <= 2".
    """
    relation = parse_relation(inequality, "inequality", allow_bare_expression=False)
    parts = list(relation.args) if isinstance(relation, sp.And) else [relation]
    if any(isinstance(p, sp.Equality) for p in parts):
        raise MathToolError("invalid_input", "this is an equation, not an inequality; use solve_equation")
    free = _relation_symbols(parts)
    var = resolve_variable(sp.Add(*free) if free else sp.Integer(0), variable)
    real_var = _real(var)
    parts = [_subs(p, {var: real_var}) for p in parts]
    var = real_var

    steps = []
    solution_set = sp.S.Reals
    for part in parts:
        expr = sp.expand(part.lhs - part.rhs)
        if _is_linear(expr, [var]):
            coefficient = expr.coeff(var)
            if coefficient.is_negative:
                steps.append(f"dividing {part} by the negative coefficient {coefficient} flips the inequality sign")
        solution_set = sp.Intersection(solution_set, sp.solveset(part, var, domain=sp.S.Reals))
    solution_set = sp.simplify(solution_set) if not isinstance(solution_set, sp.Set) else solution_set

    interval = _interval_text(solution_set)
    if solution_set == sp.EmptySet:
        text = "no solution"
    elif solution_set == sp.S.Reals:
        text = "all real numbers"
    else:
        text = _relational_text(solution_set, var)
    envelope = ok(None, result=text, steps=steps, interval=interval, variable=var,
                  solution_type="none" if solution_set == sp.EmptySet else ("all_real" if solution_set == sp.S.Reals else "set"))
    envelope["exact"] = interval
    return envelope


@math_tool
def solve_for(formula: str, variable: str) -> dict:
    """Rearrange a formula or literal equation to isolate one variable.

    Example: {"formula": "A = 1/2*b*h", "variable": "h"} -> "h = 2*A/b";
    {"formula": "y = m x + b", "variable": "x"} -> "x = (-b + y)/m".
    """
    eq = parse_equation(formula, "formula")
    var = parse_variable(variable)
    if var not in (eq.lhs - eq.rhs).free_symbols:
        raise MathToolError("invalid_input", f"variable {var} does not appear in the formula")
    solutions = [sp.simplify(s) for s in sp.solve(eq, var)]
    if not solutions:
        return ok(None, result=f"cannot isolate {var}", solution_type="none", solutions=[])
    value = solutions[0] if len(solutions) == 1 else None
    envelope = ok(value, result=_format_solutions(var, solutions), solution_type="finite", solutions=solutions,
                  restrictions=restrictions(solutions[0]) if len(solutions) == 1 else [])
    if value is None:
        envelope["exact"] = [str(s) for s in solutions]
    return envelope


@math_tool
def verify_solution(equation: str, values: dict[str, Number]) -> dict:
    """Check whether values satisfy an equation or inequality (plug back in).

    Use this after solving to catch extraneous roots.
    Example: {"equation": "sqrt(x + 2) = x", "values": {"x": -1}} -> is_solution false;
    {"equation": "2x + 3 = 11", "values": {"x": 4}} -> is_solution true.
    """
    relation = parse_relation(equation)
    if isinstance(relation, sp.And):
        parts = list(relation.args)
    else:
        parts = [relation]
    if not values:
        raise MathToolError("invalid_input", "values must map each variable to a value")
    mapping = {parse_variable(k): parse_expression(v, f"value for {k}") for k, v in values.items()}
    missing = set().union(*((p.lhs - p.rhs).free_symbols for p in parts)) - set(mapping)
    if missing:
        raise MathToolError("invalid_input", f"missing values for: {', '.join(sorted(s.name for s in missing))}")
    results = [_relation_holds(p, mapping) for p in parts]
    undefined = any(r is None for r in results)
    is_solution = not undefined and all(results)
    first = parts[0]
    lhs, rhs = sp.simplify(first.lhs.subs(mapping)), sp.simplify(first.rhs.subs(mapping))
    reason = "undefined at these values" if undefined else ("both sides agree" if is_solution else "the two sides do not agree")
    return ok(sp.true if is_solution else sp.false, result="true" if is_solution else "false",
              is_solution=is_solution, reason=reason, lhs_value=lhs, rhs_value=rhs)


TOOLS = [solve_equation, solve_quadratic, solve_system, solve_inequality, solve_for, verify_solution]
