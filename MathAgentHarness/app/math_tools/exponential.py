"""Exponent, logarithm and sequence tools (issue #8)."""

from __future__ import annotations

import sympy as sp

from .core import MathToolError, Number, math_tool, ok, parse_equation, parse_expression, resolve_variable, to_integer, to_number
from .equations import solve_equation


def _base(base: Number | None) -> sp.Expr:
    if base is None:
        return sp.Integer(10)
    if isinstance(base, str) and base.strip().lower() in ("e", "ln", "natural"):
        return sp.E
    return to_number(base, "base")


@math_tool
def log(x: Number, base: Number | None = 10) -> dict:
    """Logarithm of x in a given base (default 10; use "e" for the natural log ln).

    Example: {"x": 8, "base": 2} -> 3; {"x": 10, "base": "e"} -> "log(10)" (2.302...).
    """
    value = to_number(x, "x")
    b = _base(base)
    if not value.is_positive:
        raise MathToolError("math_error", f"log is only defined for positive numbers, got {value}")
    if not b.is_positive or b == 1:
        raise MathToolError("math_error", f"the base must be positive and not 1, got {b}")
    result = sp.simplify(sp.log(value, b)) if b != sp.E else sp.log(value)
    return ok(result, steps=[f"{b}^y = {value}"] if b != sp.E else [])


def _log_arguments(expr: sp.Expr) -> list[sp.Expr]:
    return [atom.args[0] for atom in expr.atoms(sp.log)]


@math_tool
def solve_exponential(equation: str, variable: str | None = None) -> dict:
    """Solve exponential or logarithmic equations exactly.

    Roots that make a log argument zero or negative are rejected as extraneous.
    Example: {"equation": "2^x = 32"} -> "x = 5"; {"equation": "3^(x+1) = 20"} -> "x = -1 + log(20)/log(3)";
    {"equation": "log2(x) + log2(x - 2) = 3"} -> "x = 4" (x = -2 is extraneous).
    """
    eq = parse_equation(equation)
    expr = eq.lhs - eq.rhs
    var = resolve_variable(expr, variable)
    has_exp = any(p.exp.has(var) for p in expr.atoms(sp.Pow)) or expr.has(sp.exp)
    has_log = bool(expr.atoms(sp.log))
    envelope = solve_equation.func(equation, str(var))
    if "error" in envelope:
        return envelope
    steps = list(envelope.get("steps", []))
    if has_exp:
        steps.insert(0, "take the logarithm of both sides to bring the exponent down")
    if has_log:
        steps.insert(0, "combine logarithms and rewrite in exponential form")
        arguments = _log_arguments(eq.lhs) + _log_arguments(eq.rhs)
        valid, rejected = [], list(envelope.get("extraneous", []))
        for value in envelope.get("solutions", []):
            point = parse_expression(str(value))
            if all(sp.simplify(arg.subs(var, point)).is_positive for arg in arguments):
                valid.append(value)
            else:
                rejected.append(value)
                steps.append(f"reject {var} = {value}: a log argument is not positive")
        envelope["solutions"] = valid
        envelope["extraneous"] = rejected
        if not valid:
            envelope.update(result="no real solution", solution_type="none", exact=None, decimal=None, latex=None)
        else:
            envelope["result"] = " or ".join(f"{var} = {v}" for v in valid)
    envelope["steps"] = steps
    return envelope


def _terms(rule, count: int = 5) -> list[sp.Expr]:
    return [sp.simplify(rule(k)) for k in range(1, count + 1)]


def _positive_index(n: Number | None) -> int | None:
    if n is None:
        return None
    index = int(to_integer(n, "n"))
    if index < 1:
        raise MathToolError("invalid_input", "n must be a positive integer")
    return index


@math_tool
def arithmetic_sequence(a1: Number, d: Number, n: int | None = None, term_value: Number | None = None) -> dict:
    """Arithmetic sequence a_n = a1 + (n - 1)d.

    Give n to get the n-th term and the sum of the first n terms, or term_value to
    find which term (n) has that value. Example: {"a1": 3, "d": 4, "n": 10} -> nth term 39, sum 210.
    """
    first, diff = to_number(a1, "a1"), to_number(d, "d")
    index = _positive_index(n)
    k = sp.Symbol("n")
    formula = first + (k - 1) * diff
    extra = {"formula": f"a_n = {sp.expand(formula)}", "first_terms": _terms(lambda i: formula.subs(k, i))}
    if term_value is not None:
        target = to_number(term_value, "term_value")
        if diff == 0:
            raise MathToolError("invalid_input", "with d = 0 every term equals a1, so n is not unique")
        position = sp.simplify((target - first) / diff + 1)
        is_term = bool(position.is_integer and position >= 1)
        return ok(position if is_term else None, result=f"n = {position}" if is_term else f"{target} is not a term of the sequence",
                  n=position if is_term else None, is_term=is_term, **extra)
    if index is None:
        return ok(None, result=extra["formula"], **extra)
    nth = sp.simplify(formula.subs(k, index))
    total = sp.simplify(sp.Rational(index, 2) * (first + nth))
    return ok(nth, result=f"a_{index} = {nth}, S_{index} = {total}", steps=[f"a_{index} = {first} + ({index} - 1)({diff})", f"S_{index} = {index}/2 * ({first} + {nth})"],
              nth_term=nth, sum=total, **extra)


@math_tool
def geometric_sequence(a1: Number, r: Number, n: int | None = None, term_value: Number | None = None) -> dict:
    """Geometric sequence a_n = a1 * r^(n - 1).

    Give n for the n-th term and partial sum, or term_value to find n. When |r| < 1
    the infinite sum a1 / (1 - r) is also returned.
    Example: {"a1": 2, "r": 3, "n": 5} -> nth term 162, sum 242.
    """
    first, ratio = to_number(a1, "a1"), to_number(r, "r")
    if first == 0 or ratio == 0:
        raise MathToolError("invalid_input", "a1 and r must be non-zero for a geometric sequence")
    index = _positive_index(n)
    k = sp.Symbol("n")
    formula = first * ratio ** (k - 1)
    extra = {
        "formula": f"a_n = {formula}",
        "first_terms": _terms(lambda i: formula.subs(k, i)),
        "infinite_sum": sp.simplify(first / (1 - ratio)) if abs(ratio) < 1 else None,
        "converges": bool(abs(ratio) < 1),
    }
    if term_value is not None:
        target = to_number(term_value, "term_value")
        candidates = [i for i in range(1, 200) if sp.simplify(formula.subs(k, i) - target) == 0]
        is_term = bool(candidates)
        return ok(sp.Integer(candidates[0]) if is_term else None,
                  result=f"n = {candidates[0]}" if is_term else f"{target} is not a term of the sequence",
                  n=candidates[0] if is_term else None, is_term=is_term, **extra)
    if index is None:
        return ok(None, result=extra["formula"], **extra)
    nth = sp.simplify(formula.subs(k, index))
    total = first * index if ratio == 1 else sp.simplify(first * (1 - ratio**index) / (1 - ratio))
    return ok(nth, result=f"a_{index} = {nth}, S_{index} = {total}", nth_term=nth, sum=total, **extra)


@math_tool
def sequence_from_terms(terms: dict[str, Number], kind: str = "auto") -> dict:
    """Find a sequence's rule from known terms, keyed by position.

    kind is "arithmetic", "geometric" or "auto" (auto needs 3+ consecutive terms).
    Example: {"terms": {"1": 2, "4": 11}, "kind": "arithmetic"} -> d = 3, a1 = 2;
    {"terms": {"1": 3, "2": 6, "3": 12}} -> geometric, r = 2.
    """
    if not terms or len(terms) < 2:
        raise MathToolError("invalid_input", "give at least two known terms, e.g. {\"1\": 2, \"4\": 11}")
    known = sorted(((int(to_integer(k, "position")), to_number(v, f"term {k}")) for k, v in terms.items()))
    if any(position < 1 for position, _ in known):
        raise MathToolError("invalid_input", "positions start at 1")
    kind = kind.lower().strip()
    if kind == "auto":
        values = [v for _, v in known]
        consecutive = [p for p, _ in known] == list(range(known[0][0], known[0][0] + len(known)))
        if len(known) < 3 or not consecutive:
            raise MathToolError("invalid_input", "auto-detection needs 3 or more consecutive terms; otherwise pass kind")
        differences = {sp.simplify(b - a) for a, b in zip(values, values[1:])}
        ratios = {sp.simplify(b / a) for a, b in zip(values, values[1:])} if all(v != 0 for v in values) else set()
        if len(differences) == 1:
            kind = "arithmetic"
        elif len(ratios) == 1:
            kind = "geometric"
        else:
            return ok(None, result="neither arithmetic nor geometric", kind="neither")
    (m, am), (p, ap) = known[0], known[1]
    if kind == "arithmetic":
        d = sp.simplify((ap - am) / (p - m))
        a1 = sp.simplify(am - (m - 1) * d)
        return ok(None, result=f"arithmetic: a1 = {a1}, d = {d}", kind=kind, a1=a1, d=d, formula=f"a_n = {sp.expand(a1 + (sp.Symbol('n') - 1) * d)}")
    if kind == "geometric":
        if am == 0:
            raise MathToolError("invalid_input", "terms of a geometric sequence cannot be 0")
        gap = p - m
        x = sp.Symbol("r", real=True)
        ratios = [sp.nsimplify(s) for s in sp.solve(x**gap - ap / am, x)]
        options = [{"r": r, "a1": sp.simplify(am / r ** (m - 1))} for r in sorted(ratios, key=lambda v: -float(sp.N(v)))]
        if not options:
            return ok(None, result="no real common ratio", kind=kind, options=[])
        best = options[0]
        return ok(None, result=f"geometric: a1 = {best['a1']}, r = {best['r']}" + (" (r could also be negative)" if len(options) > 1 else ""),
                  kind=kind, a1=best["a1"], r=best["r"], options=options)
    raise MathToolError("invalid_input", "kind must be 'arithmetic', 'geometric' or 'auto'")


TOOLS = [log, solve_exponential, arithmetic_sequence, geometric_sequence, sequence_from_terms]
